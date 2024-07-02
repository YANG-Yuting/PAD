import datasets
import numpy as np
import torch
import os
import time

"""from shield"""
from shield.model import BertClassifierDARTS, RobertaClassifierDARTS
from shield.utils import get_diversity_training_term

"""from mine"""
from data_utils import read_text, dataset_mapping_trec, MyDataset
from transformers import AutoTokenizer
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process
from config import args
from models import eval_model

CEloss = torch.nn.CrossEntropyLoss()


device = 'cuda:0'
grad_clip = 3
patience = 2
model_types = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base'}
model_type = model_types[args.target_model]
training_temp = 1.0
alpha_darts = 0.5


def train_epoch(epoch, best_test, model, opt, opt_decision, dataloader_train, dataloader_test, tokenizer):
    """Train for a epoch"""
    time_start = time.time()
    model.train()
    cnt = 0
    for idx, (*train_x, train_y) in enumerate(dataloader_train):
        opt.zero_grad()
        input_ids, input_mask, segment_ids, train_y = train_x[0].cuda(args.rank), train_x[1].cuda(args.rank), train_x[2].cuda(args.rank), train_y.cuda(args.rank)
        cnt += 1
        if torch.cuda.device_count() > 1:
            preds = model.module(input_ids, input_mask)
        else:
            preds = model(input_ids, input_mask)
        preds_prob = torch.nn.functional.softmax(preds, dim=-1)
        loss = CEloss(preds_prob, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        opt_decision.zero_grad()
        for *val_x, val_y in dataloader_test:  # 取验证集的第一个batch作为验证
            continue
        input_ids_val, input_mask_val, segment_ids_val, val_y = val_x[0].cuda(args.rank), val_x[1].cuda(args.rank), val_x[2].cuda(args.rank), val_y.cuda(args.rank)
        if torch.cuda.device_count() > 1:
            preds_val = model.module(input_ids_val, input_mask_val)
        else:
            preds_val = model(input_ids_val, input_mask_val)
        preds_prob_val = torch.nn.functional.softmax(preds_val, dim=-1)
        val_loss = CEloss(preds_prob_val, val_y)
        batch_val = {}
        batch_val['labels'] = val_y
        reg_diversity_training, reg_diff = get_diversity_training_term(model, batch_val, logsumexp=False)
        reg_term = torch.tensor(alpha_darts).to(device) * (reg_diversity_training) - torch.tensor(alpha_darts).to(device) * reg_diff
        val_loss = val_loss + reg_term
        val_loss.backward()
        opt_decision.step()

    # For process 0
    if is_main_process() and epoch % 1 == 0:
        model.eval()
        model.inference = True
        num, correct = 0, 0
        for *test_x, test_y in dataloader_test:
            input_ids_test, input_mask_test, segment_ids_test, test_y = test_x[0].cuda(args.rank), test_x[1].cuda(args.rank), test_x[2].cuda(args.rank), test_y.cuda(args.rank)
            preds = model(input_ids_test, input_mask_test)
            preds_prob = torch.nn.functional.softmax(preds, dim=-1)
            correct += torch.sum(preds_prob.argmax(dim=-1) == test_y).item()
            num += test_y.size()[0]
        test_acc = correct/num
        model.inference = False
        time_end = time.time()
        time_used = time_end - time_start
        print("Epoch={} time={:.2f}s loss={:.6f} val_loss={:.6f} test_acc={:.6f}".format( epoch, time_used, loss.mean().item(), val_loss.mean().item(), test_acc))
        if test_acc > best_test:
            best_test = test_acc
            if not os.path.exists(args.save_path): os.makedirs(args.save_path)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.save_path + '/pytorch_model.bin')
                if args.target_model == 'bert':
                    model.module.bert_layer.config.to_json_file(args.save_path + '/config.json')
                else:
                    model.module.roberta_layer.config.to_json_file(args.save_path + '/config.json')
            else:
                torch.save(model.state_dict(), args.save_path + '/pytorch_model.bin')
                if args.target_model == 'bert':
                    model.bert_layer.config.to_json_file(args.save_path + '/config.json')
                else:
                    model.roberta_layer.config.to_json_file(args.save_path + '/config.json')

            tokenizer.save_vocabulary(args.save_path)
            print('save model when test acc=', test_acc)

    # 保存当前epoch
    if is_main_process():
        save_path = args.save_path + '_final'
        if not os.path.exists(save_path): os.makedirs(save_path)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), save_path + '/pytorch_model.bin')
            if args.target_model == 'bert':
                model.module.bert_layer.config.to_json_file(save_path + '/config.json')
            else:
                model.module.roberta_layer.config.to_json_file(save_path + '/config.json')
        else:
            torch.save(model.state_dict(), save_path + '/pytorch_model.bin')
            if args.target_model == 'bert':
                model.bert_layer.config.to_json_file(save_path + '/config.json')
            else:
                model.roberta_layer.config.to_json_file(save_path + '/config.json')

        tokenizer.save_vocabulary(save_path)

    dist.barrier()  # 这一句作用是：所有进程(gpu)上的代码都执行到这，才会执行该句下面的代码
    return best_test


def main():

    """Setting for parallel training"""
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # args.rank = 0
    init_distributed_mode(args=args)
    torch.cuda.set_device(args.rank)

    """Load data"""
    if args.task == 'trec':
        test_dataset = datasets.load_dataset("trec", split="test[:500]").map(lambda x: dataset_mapping_trec(x))
        train_dataset = datasets.load_dataset("trec", split="train").map(lambda x: dataset_mapping_trec(x))
        train_data_list, train_label_list, test_data_list, test_label_list = [], [], [], []
        for d in train_dataset:
            train_data_list.append(d['x'])
            train_label_list.append(d['y'])
        for d in test_dataset:
            test_data_list.append(d['x'])
            test_label_list.append(d['y'])
    else:
        # Read from files
        train_data_list, train_label_list, test_data_list, test_label_list = read_text(args.data_path, args.task)
        # train_data_list, train_label_list = train_data_list[:1000], train_label_list[:1000]
        # 随机选取1000个测试集作为验证集（从200个往后取，这样与攻击数据无重合）
        if args.task == 'imdb':
            test_x, test_y = test_data_list[200:], test_label_list[200:]
            c = list(zip(test_x, test_y))
            np.random.seed(15)
            np.random.shuffle(c)
            test_x, test_y = zip(*c)
            test_data_list, test_label_list = test_x[:1000], test_y[:1000]

    """Initialize model"""
    if args.target_model == 'bert':
        model = BertClassifierDARTS(model_type=model_type, freeze_bert=True, output_dim=args.num_labels, ensemble=1, N=args.num_models, temperature=training_temp,
                                gumbel=1, scaler=1, darts=True, device=device)
    else:
        model = RobertaClassifierDARTS(model_type=model_type, freeze_roberta=True, output_dim=args.num_labels, ensemble=1, N=args.num_models, temperature=training_temp,
                                gumbel=1, scaler=1, darts=True, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

    """Prepare for training"""
    # build dataloader
    if args.task != 'snli':
        dataset_train, _ = MyDataset(args, tokenizer).transform_text(train_data_list, train_label_list)
        if torch.cuda.device_count() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
            nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)
        else:
            train_sampler = torch.utils.data.SequentialSampler(dataset_train)
            dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size)
    dataset_test, _ = MyDataset(args, tokenizer).transform_text(test_data_list, test_label_list)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, sampler=test_sampler, batch_size=args.batch_size)

    if is_main_process():
        print('#Train data:', len(train_data_list))
        para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : {:4f}M params'.format(model._get_name(), para * 4 / 1000 / 1000))

    """Training..."""
    parameters = filter(lambda p: 'heads' in p[0], model.named_parameters())
    opt = torch.optim.Adam([p[1] for p in parameters], lr=3e-5)

    decision_parameters = filter(lambda p: 'darts_decision' in p[0], model.named_parameters())
    opt_decision = torch.optim.Adam([p[1] for p in decision_parameters], lr=0.1)

    epoch = 100
    best_test = 0
    for e in range(epoch):
        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        if args.task == 'snli':
            # 每个训练epoch，随机选取10000个训练数据
            c = list(zip(train_data_list, train_label_list))
            # np.random.seed(15)
            np.random.shuffle(c)
            train_x, train_y = zip(*c)
            train_data_list, train_label_list = train_x[:10000], train_y[:10000]
            dataset_train, _ = MyDataset(args, tokenizer).transform_text(train_data_list, train_label_list)
            if torch.cuda.device_count() > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
                nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
                dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)
            else:
                train_sampler = torch.utils.data.SequentialSampler(dataset_train)
                dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size)
        best_test = train_epoch(e, best_test, model, opt, opt_decision, dataloader_train, dataloader_test, tokenizer)


if __name__ == '__main__':
    main()




