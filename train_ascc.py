import torch
import torch.nn as nn
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.optim as optim
import time

# parallel training
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process

from config import args
import numpy as np
import datasets
from transformers import AutoTokenizer, AutoConfig, RobertaForSequenceClassification, BertForSequenceClassification
from data_utils import read_text, dataset_mapping, dataset_mapping_trec, MyDataset
from models import eval_model
import torch.nn.functional as F

if args.task in['mr', 'trec', 'snli']:
    weight_kl = 0.1
    weight_adv = 0.5
    kl_control = 1
    sparse_weight = 15
elif args.task == 'imdb':
    # imdb 用mr的参数 clean acc掉很多，因此降低weight_adv和weight_kl
    weight_kl = 0.01
    weight_adv = 0.1
    kl_control = 1
    sparse_weight = 15

# trec 没效果 换权重试试
# weight_kl = 1
# weight_adv = 1
# kl_control = 1
# sparse_weight = 1

criterion = nn.CrossEntropyLoss()

"""当input_embeds非空时，其作为word embedding使用"""

criterion_kl = nn.KLDivLoss(reduction="sum")


class ASCCBERT(BertForSequenceClassification):
    def __init__(self, config):
        super(ASCCBERT, self).__init__(config)
        self.inference = False

    """ASCC: adv embedding来自在凸包中的搜索"""
    def get_adv_ascc(self, input_ids, input_mask, segment_ids, syn_input_ids, syn_valid):
        if self.inference:
            self.eval()
        else:
            self.train()

        batch_size, text_len = input_ids.shape
        w = torch.empty(batch_size, text_len, args.syn_num, 1).cuda(args.rank)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()

        params = [w]
        optimizer_max = torch.optim.Adam(params, lr=10, weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, args.syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, args.syn_num, 1)-1)
            return F.softmax(ww, dim=-2)

        def get_comb_ww(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, args.syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, args.syn_num, 1)-1)
            return ww

        def get_comb(p, syn):
            return (p* syn.detach()).sum(-2)

        with torch.no_grad():
            logit_ori = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).logits
        if torch.cuda.device_count() > 1:
            word_embedding_syn = model.module.bert.embeddings.word_embeddings(syn_input_ids)  # [b_size*syn_num, seq_len, h_size]
        else:
            word_embedding_syn = model.bert.embeddings.word_embeddings(syn_input_ids)  # [b_size*syn_num, seq_len, h_size]

        num_steps = 3
        loss_sum = 0.0
        for _ in range(num_steps):
            optimizer_max.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, syn_valid)  # [b_size, seq_len, syn_num , 1]
                embd_adv = get_comb(F.softmax(ww, dim=-2), word_embedding_syn.reshape(batch_size, text_len, args.syn_num, -1))  # [b_size, seq_len, h_size]
                if torch.cuda.device_count() > 1:
                    logit_adv = model.module(token_type_ids=segment_ids, attention_mask=input_mask, inputs_embeds=embd_adv).logits
                else:
                    logit_adv = model(token_type_ids=segment_ids, attention_mask=input_mask, inputs_embeds=embd_adv).logits

                loss = -criterion_kl(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori.detach(), dim=1))
                loss_sparse = (-F.softmax(ww, dim=-2) * F.log_softmax(ww, dim=-2)).sum(-2).mean()
                loss = loss + sparse_weight * loss_sparse
            loss.backward()
            optimizer_max.step()
            loss_sum += loss.item()
        comb_p = get_comb_p(w, syn_valid)
        # print(w)
        # print(syn_valid)
        # print(input_ids)
        # print(syn_input_ids)
        # if input_ids[0].equal(syn_input_ids[0]):
        #     print('llllll')
        # exit(0)
        return comb_p.detach(), loss_sum / num_steps


class ASCCROBERTA(RobertaForSequenceClassification):
    def __init__(self, config):
        super(ASCCROBERTA, self).__init__(config)
        self.inference = False

    """ASCC: adv embedding来自在凸包中的搜索"""
    def get_adv_ascc(self, input_ids, input_mask, segment_ids, syn_input_ids, syn_valid):
        if self.inference:
            self.eval()
        else:
            self.train()

        batch_size, text_len = input_ids.shape
        w = torch.empty(batch_size, text_len, args.syn_num, 1).cuda(args.rank)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()

        params = [w]
        optimizer_max = torch.optim.Adam(params, lr=10, weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, args.syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, args.syn_num, 1)-1)
            return F.softmax(ww, dim=-2)

        def get_comb_ww(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, args.syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, args.syn_num, 1)-1)
            return ww

        def get_comb(p, syn):
            return (p* syn.detach()).sum(-2)

        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                logit_ori = model.module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).logits
            else:
                logit_ori = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).logits
        if torch.cuda.device_count() > 1:
            word_embedding_syn = model.module.roberta.embeddings.word_embeddings(syn_input_ids)  # [b_size*syn_num, seq_len, h_size]
        else:
            word_embedding_syn = model.roberta.embeddings.word_embeddings(syn_input_ids)  # [b_size*syn_num, seq_len, h_size]

        num_steps = 3
        loss_sum = 0.0
        for _ in range(num_steps):
            optimizer_max.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, syn_valid)  # [b_size, seq_len, syn_num , 1]
                embd_adv = get_comb(F.softmax(ww, dim=-2), word_embedding_syn.reshape(batch_size, text_len, args.syn_num, -1))  # [b_size, seq_len, h_size]
                if torch.cuda.device_count() > 1:
                    logit_adv = model.module(token_type_ids=segment_ids, attention_mask=input_mask, inputs_embeds=embd_adv).logits
                else:
                    logit_adv = model(token_type_ids=segment_ids, attention_mask=input_mask, inputs_embeds=embd_adv).logits

                loss = -criterion_kl(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori.detach(), dim=1))
                loss_sparse = (-F.softmax(ww, dim=-2) * F.log_softmax(ww, dim=-2)).sum(-2).mean()
                loss = loss + sparse_weight * loss_sparse
            loss.backward()
            optimizer_max.step()
            loss_sum += loss.item()
        comb_p = get_comb_p(w, syn_valid)
        return comb_p.detach(), loss_sum / num_steps


def train_epoch(epoch, best_test, model, optimizer, dataloader_train, dataloader_train_syn, dataloader_test, dataloader_test_syn):
    time_start = time.time()
    model.train()

    for idx, (data, data_syn) in enumerate(zip(dataloader_train, dataloader_train_syn)):
        optimizer.zero_grad()
        for t in range(len(data)):
            data[t] = data[t].cuda(args.rank)
        for t in range(len(data_syn)):
            data_syn[t] = data_syn[t].cuda(args.rank)
        input_ids, input_mask, segment_ids, train_y = data
        syn_input_ids, syn_input_mask, syn_segment_ids, syn_flags, syn_labels = data_syn

        """1. Maximize in convex hull"""
        model.inference = False
        if torch.cuda.device_count() > 1:
            adv_comb_p, loss_maxOP = model.module.get_adv_ascc(input_ids, input_mask, segment_ids, syn_input_ids, syn_flags)  # [b_size,seq_len,syn_num,1]
        else:
            adv_comb_p, loss_maxOP = model.get_adv_ascc(input_ids, input_mask, segment_ids, syn_input_ids, syn_flags)  # [b_size,seq_len,syn_num,1]

        """2. Minimize """
        if torch.cuda.device_count() > 1:
            logit_ori = model.module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).logits
        else:
            logit_ori = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).logits

        # preds_ori = torch.argmax(logit_ori, dim=-1)
        loss_clean = criterion(F.softmax(logit_ori, dim=1), train_y.cuda(args.rank))
        batch_size, text_len = input_ids.shape
        if args.target_model == 'bert':
            if torch.cuda.device_count() > 1:
                word_embedding_syn = model.module.bert.embeddings.word_embeddings(syn_input_ids).reshape(batch_size, text_len, args.syn_num, -1)
            else:
                word_embedding_syn = model.bert.embeddings.word_embeddings(syn_input_ids).reshape(batch_size, text_len, args.syn_num, -1)
        elif args.target_model == 'roberta':
            if torch.cuda.device_count() > 1:
                word_embedding_syn = model.module.roberta.embeddings.word_embeddings(syn_input_ids).reshape(batch_size, text_len, args.syn_num, -1)
            else:
                word_embedding_syn = model.roberta.embeddings.word_embeddings(syn_input_ids).reshape(batch_size, text_len, args.syn_num, -1)


        emb_adv = (adv_comb_p * word_embedding_syn).sum(-2)  # [batch_size, seq_len, h_size]
        if torch.cuda.device_count() > 1:
            logit_adv = model.module(token_type_ids=segment_ids, attention_mask=input_mask, inputs_embeds=emb_adv).logits
        else:
            logit_adv = model(token_type_ids=segment_ids, attention_mask=input_mask, inputs_embeds=emb_adv).logits

        loss_adv = criterion(F.softmax(logit_adv, dim=1), train_y.cuda(args.rank))
        criterion_kl = nn.KLDivLoss(reduction="sum")
        loss_kl = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

        loss = loss_clean + weight_adv * loss_adv + weight_kl * kl_control * loss_kl
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


    # For process 0
    if is_main_process() and epoch % 1 == 0:
        test_acc = eval_model(model, dataloader_test)
        time_end = time.time()
        time_used = time_end - time_start
        print("Epoch={} time={:.2f}s loss_maxOP={:.6f} loss_minOP={:.6f} loss_kl={:.6f} loss_adv={:.6f} loss_clean={:.6f} test_acc={:.6f}".format(
        epoch, time_used, loss_maxOP, loss.mean().item(), loss_kl.mean().item(), loss_adv.mean().item(), loss_clean.mean().item(), test_acc))
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


if __name__ == '__main__':

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
        # train_data_list, train_label_list = train_data_list[:2], train_label_list[:2]
        # test_data_list, test_label_list = test_data_list[:10], test_label_list[:10]
        # 随机选取1000个测试集作为验证集（从200个往后取，这样与攻击数据无重合）
        if args.task == 'imdb':
            test_x, test_y = test_data_list[200:], test_label_list[200:]
            c = list(zip(test_x, test_y))
            np.random.seed(15)
            np.random.shuffle(c)
            test_x, test_y = zip(*c)
            test_data_list, test_label_list = test_x[:1000], test_y[:1000]

    """Initialize model"""
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    config = AutoConfig.from_pretrained(args.target_model_path)
    config.perturb_attention = False
    model_types = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base'}
    model_type = model_types[args.target_model]
    if args.target_model_path == model_type:
        if args.target_model == 'bert':
            model = ASCCBERT.from_pretrained(model_type, num_labels=args.num_labels).cuda(args.rank)
        elif args.target_model == 'roberta':
            model = ASCCROBERTA.from_pretrained(model_type, num_labels=args.num_labels).cuda(args.rank)
    else:
        checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
        if args.target_model == 'bert':
            model = ASCCBERT.from_pretrained(model_type, state_dict=checkpoint, config=config).cuda(args.rank)
        elif args.target_model == 'roberta':
            model = ASCCROBERTA.from_pretrained(model_type, state_dict=checkpoint, config=config).cuda(args.rank)

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

    """Prepare for training"""
    # build dataloader
    if args.task != 'snli' and args.task != 'imdb':
        dataset_train, dataset_train_syn = MyDataset(args, tokenizer).transform_text(train_data_list, train_label_list)
        if torch.cuda.device_count() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
            train_sampler_syn = torch.utils.data.distributed.DistributedSampler(dataset_train_syn)
            nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)
            dataloader_train_syn = torch.utils.data.DataLoader(dataset_train_syn, sampler=train_sampler_syn, pin_memory=True, num_workers=nw, batch_size=args.batch_size * args.syn_num)
        else:
            train_sampler = torch.utils.data.SequentialSampler(dataset_train)
            train_sampler_syn = torch.utils.data.SequentialSampler(dataset_train_syn)
            dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size)
            dataloader_train_syn = torch.utils.data.DataLoader(dataset_train_syn, sampler=train_sampler_syn, batch_size=args.batch_size * args.syn_num)

    dataset_test, dataset_test_syn = MyDataset(args, tokenizer).transform_text(test_data_list, test_label_list)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, sampler=test_sampler, batch_size=args.batch_size)
    test_sampler_syn = torch.utils.data.SequentialSampler(dataset_test_syn)
    dataloader_test_syn = torch.utils.data.DataLoader(dataset_test_syn, sampler=test_sampler_syn, batch_size=args.batch_size * args.syn_num)

    if is_main_process():
        print('#Train data:', len(train_data_list))
        para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : {:4f}M params'.format(model._get_name(), para * 4 / 1000 / 1000))
        # test_acc = eval_model(model, dataloader_test)
        # print('Original test acc = %.4f' % test_acc)
        # exit(0)

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=args.lr)
    best_test = 0
    num_epochs = 100

    for epoch in range(num_epochs):
        if args.task == 'snli' or args.task == 'imdb':
            # 每个训练epoch，随机选取10000个训练数据
            c = list(zip(train_data_list, train_label_list))
            # np.random.seed(15)
            np.random.shuffle(c)
            train_x, train_y = zip(*c)
            train_data_list, train_label_list = train_x[:10000], train_y[:10000]
            dataset_train, dataset_train_syn = MyDataset(args, tokenizer).transform_text(train_data_list, train_label_list)
            if torch.cuda.device_count() > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
                train_sampler_syn = torch.utils.data.distributed.DistributedSampler(dataset_train_syn)
                nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
                dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, pin_memory=True, num_workers=nw, batch_size=args.batch_size)
                dataloader_train_syn = torch.utils.data.DataLoader(dataset_train_syn, sampler=train_sampler_syn, pin_memory=True, num_workers=nw, batch_size=args.batch_size * args.syn_num)
            else:
                train_sampler = torch.utils.data.SequentialSampler(dataset_train)
                train_sampler_syn = torch.utils.data.SequentialSampler(dataset_train_syn)
                dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size)
                dataloader_train_syn = torch.utils.data.DataLoader(dataset_train_syn, sampler=train_sampler_syn, batch_size=args.batch_size * args.syn_num)

        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        best_test = train_epoch(epoch, best_test, model, optimizer, dataloader_train, dataloader_train_syn, dataloader_test, dataloader_test_syn)

