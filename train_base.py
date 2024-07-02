import datasets
import numpy as np
import torch
import os, time
import torch.nn as nn
import copy
import torch.nn.functional as F
import json

"""from mine"""
from data_utils import read_text, dataset_mapping, dataset_mapping_trec, MyDataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process
from models import eval_model
from config import args
criterion = torch.nn.CrossEntropyLoss()


def train_epoch(epoch, best_test, model, optimizer, dataloader_train, dataloader_test, tokenizer):
    """Train for a epoch"""
    time_start = time.time()
    model.train()
    cnt = 0
    for idx, (*train_x, train_y) in enumerate(dataloader_train):
        optimizer.zero_grad()
        input_ids, input_mask, segment_ids, train_y = train_x[0].cuda(args.rank), train_x[1].cuda(args.rank), train_x[2].cuda(args.rank), train_y.cuda(args.rank)
        cnt += 1

        if torch.cuda.device_count() > 1:
            if args.target_model == 'bart':
                outputs = model.module(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = model.module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        else:
            if args.target_model == 'bart':
                outputs = model(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        logits = outputs.logits
        # preds_ori = torch.argmax(logit_ori, dim=-1)
        loss = criterion(F.softmax(logits, dim=1), train_y)
        loss = loss.mean()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
    lr_decay = 1.0
    if lr_decay > 0:
        optimizer.param_groups[0]['lr'] *= lr_decay

    # For process 0
    if is_main_process() and epoch % 1 == 0:
        test_acc = eval_model(model, dataloader_test)
        time_end = time.time()
        time_used = time_end - time_start
        print("Epoch={} time={:.2f}s train_loss={:.6f} test_acc={:.6f}".format(epoch, time_used, loss.item(), test_acc))
        if test_acc > best_test:
            best_test = test_acc
            if not os.path.exists(args.save_path): os.makedirs(args.save_path)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.save_path + '/pytorch_model.bin')
                model.module.config.to_json_file(args.save_path + '/config.json')
            else:
                torch.save(model.state_dict(), args.save_path + '/pytorch_model.bin')
                model.config.to_json_file(args.save_path + '/config.json')
            tokenizer.save_vocabulary(args.save_path)
            print('save model when test acc=', test_acc)

    # time_end = time.time()
    # time_used = time_end - time_start
    # 保存当前epoch
    if is_main_process():
        save_path = args.save_path + '_final'
        if not os.path.exists(save_path): os.makedirs(save_path)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), save_path + '/pytorch_model.bin')
            model.module.config.to_json_file(save_path + '/config.json')
        else:
            torch.save(model.state_dict(), save_path + '/pytorch_model.bin')
            model.config.to_json_file(save_path + '/config.json')
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
        print('#Train data', len(train_label_list))
        # train_data_list, train_label_list = train_data_list[:3], train_label_list[:3]
        # test_data_list, test_label_list = test_data_list[:2], test_label_list[:2]
        # 随机选取1000个测试集作为验证集（从200个往后取，这样与攻击数据无重合）
        if args.task == 'imdb':
            test_x, test_y = test_data_list[200:], test_label_list[200:]
            c = list(zip(test_x, test_y))
            np.random.seed(15)
            np.random.shuffle(c)
            test_x, test_y = zip(*c)
            test_data_list, test_label_list = test_x[:1000], test_y[:1000]
        if args.attack_alg == 'all':
            # 综合3个level的攻击
            adv_data_list, adv_label_list = [], []
            for attack_alg in ['scpn','deepwordbug', 'bertattack']:
                adv_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/train_set/%s_%s_%s.json' % (
                args.task, args.target_model, attack_alg)
                try:
                    advs = json.load(open(adv_path, 'r'))
                    for i in range(len(advs)):
                        adv = advs[i]
                        adv_data_list.append(adv['adv_text'])
                        adv_label_list.append(adv['label'])
                except:
                    advs = open(adv_path, 'r').readlines()
                    for i in range(len(advs)):
                        adv = json.loads(advs[i])
                        adv_data_list.append(adv['adv_text'])
                        adv_label_list.append(adv['label'])
        elif args.attack_alg is not None:
            # 加载对抗样本
            adv_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/train_set/%s_%s_%s.json' % (args.task, args.target_model, args.attack_alg)
            adv_data_list, adv_label_list = [], []
            try:
                advs = json.load(open(adv_path, 'r'))
                for i in range(len(advs)):
                    adv = advs[i]
                    adv_data_list.append(adv['adv_text'])
                    adv_label_list.append(adv['label'])
            except:
                advs = open(adv_path, 'r').readlines()
                for i in range(len(advs)):
                    adv = json.loads(advs[i])
                    adv_data_list.append(adv['adv_text'])
                    adv_label_list.append(adv['label'])

        # print('#Used adversarial examples:', len(adv_label_list))
        # train_data_list.extend(adv_data_list)
        # train_label_list.extend(adv_label_list)
        # print('#Final train data', len(train_label_list))

    """Initialize model"""
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels).cuda(args.rank)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

    """Prepare for training"""
    if args.task != 'snli':
        dataset_train, _ = MyDataset(args, tokenizer).transform_text(train_data_list, train_label_list)
        if torch.cuda.device_count() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
            nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)
        else:
            train_sampler = torch.utils.data.SequentialSampler(dataset_train)
            dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size)
    # build dataloader
    dataset_test, _ = MyDataset(args, tokenizer).transform_text(test_data_list, test_label_list)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, sampler=test_sampler, batch_size=args.batch_size)
    # 关闭部分的参数更新

    if is_main_process():
        print('#Train data:', len(train_data_list))
        para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : {:4f}M params'.format(model._get_name(), para / 1000 / 1000))
        # exit(0)
        # test_acc = eval_model(model, dataloader_test)
        # print('Original test acc = %.4f' % test_acc)
        # exit(0)

    """Training..."""
    need_grad = lambda x: x.requires_grad
    optimizer = torch.optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
    epoch = 200  # 10
    best_test = 0
    for e in range(epoch):
        if args.task == 'snli':
            # 每个训练epoch，随机选取10000个训练数据
            # c = list(zip(train_data_list, train_label_list))
            # # np.random.seed(15)
            # np.random.shuffle(c)
            # train_x, train_y = zip(*c)
            # train_data_list, train_label_list = train_x[:10000], train_y[:10000]

            dataset_train, _ = MyDataset(args, tokenizer).transform_text(train_data_list, train_label_list)
            if torch.cuda.device_count() > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
                nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
                dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)
            else:
                train_sampler = torch.utils.data.SequentialSampler(dataset_train)
                dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size)
        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        best_test = train_epoch(e, best_test, model, optimizer, dataloader_train, dataloader_test, tokenizer)


if __name__ == '__main__':
    main()




