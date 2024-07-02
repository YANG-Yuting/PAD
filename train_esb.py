import datasets
import numpy as np
import torch
import os, time
import torch.nn as nn
import copy
# import multiprocessing as mp
# mp.set_start_method('spawn')


"""from mine"""
from data_utils import read_text, dataset_mapping, dataset_mapping_trec, MyDataset
import transformers
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process
from models import EnsembleBERT, EnsembleROBERTA
from config import args
from models import eval_model



# lamda = 1
# log_det_lamda = 0.1
log_offset = 1e-8 # 1e-20
norm_offset = 1e-8  # 我加入的
det_offset = 1e-6
CEloss = torch.nn.CrossEntropyLoss()


## Function ##
def Entropy(input):
    #input shape is batch_size X num_class
    ent = torch.sum(-torch.mul(input, torch.log(input + log_offset)), dim=-1)
    return ent
    # return tf.reduce_sum(-tf.multiply(input, tf.log(input + log_offset)), axis=-1)


def Ensemble_Entropy(y_pred, num_model=args.num_models):
    chunk_size = int(y_pred.size()[0] / num_model)
    y_p = torch.split(y_pred, split_size_or_sections=chunk_size, dim=0)
    y_p_all = 0
    for i in range(num_model):
        y_p_all += y_p[i]
    Ensemble = Entropy(y_p_all / num_model)
    return Ensemble


def log_det(y_pred, num_model=args.num_models):  # y_true为标签值

    # bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, 0)  # batch_size X (num_class X num_models), 2-D
    # mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
    # mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, args.num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
    # mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
    # matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
    # all_log_det = tf.linalg.logdet(matrix + det_offset * tf.expand_dims(tf.eye(num_model), 0)) # batch_size X 1, 1-D

    # if args.num_labels > 2:
    #     y_true = F.one_hot(y_true, args.num_labels)
    #     bool_R_y_true = torch.ne(torch.ones_like(y_true) - y_true, 0)  # 标记y_true为0的地方为True（过滤实际label对应位置）
    #     mask_non_y_pred = torch.masked_select(y_pred, bool_R_y_true)  # 保留除了实际label以外其他label的预测概率
    #     mask_non_y_pred = torch.reshape(mask_non_y_pred, [-1, num_model, args.num_labels - 1])
    #     mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, dim=2, keepdim=True)  # 对于二分类，此时mask_non_pred里只剩一个label，因此二范数仍然是对于该label的预测值本身，因此除完之后是1
    #     matrix = torch.matmul(mask_non_y_pred, torch.transpose(mask_non_y_pred, 2, 1))
    #     all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    # else:
    #     y_pred = torch.reshape(y_pred, [-1, num_model, args.num_labels])
    #     y_pred_norm = y_pred / torch.norm(y_pred, dim=2, keepdim=True)
    #     matrix = torch.matmul(y_pred_norm, torch.transpose(y_pred_norm, 2, 1))
    #     all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))

    b_size = int(y_pred.size()[0]/num_model)

    # y_pred_new=y_pred.reshape(num_model,b_size,y_pred.size()[1])
    # y_pred_new=y_pred_new.permute(1,0,2).contiguous()
    # matrix = torch.matmul(y_pred_new, torch.transpose(y_pred_new, 2, 1))
    # all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    # return all_log_det

    y_pred_new = []
    for b in range(b_size):
        y_pred_asent = y_pred[b::b_size]  # num_models, n_labels
        y_pred_new.append(y_pred_asent)
    y_pred_new = torch.cat(y_pred_new, dim=0).cuda(args.rank)
    y_pred_new = y_pred_new.reshape(b_size, args.num_models, -1)
    # print(y_pred_new.size(), y_pred_new)
    matrix = torch.matmul(y_pred_new, torch.transpose(y_pred_new, 2, 1)).cuda(args.rank)  # num_models,num_models
    # print(matrix)
    all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    # print(all_log_det)
    return all_log_det


def Ensemble_Entropy_hp(attention_probs, num_model=args.num_models):

    b_size = attention_probs[0].size()[0]
    num_heads = attention_probs[0].size()[1]
    seq_len = attention_probs[0].size()[2]
    attention_probs = torch.stack(attention_probs)

    a_mask = torch.eye(seq_len, dtype=torch.bool).cuda(args.rank)
    a_mask = torch.where(a_mask == False, True, False)

    attention_probs = torch.sum(attention_probs, dim=2)

    mask_a_probs = torch.masked_select(attention_probs, a_mask)

    mask_a_probs = mask_a_probs.reshape(num_model, -1, seq_len - 1)   #[modle_num,da,seq_len]

    mask_a_probs=mask_a_probs/torch.sum(mask_a_probs,dim=-1,keepdim=True)

    Ensemble=Entropy(torch.mean(mask_a_probs,dim=0))
    Ensemble=Ensemble.reshape(b_size,-1)
    Ensemble=torch.mean(Ensemble,dim=-1)
    return Ensemble

    #attention_det = mask_a_probs.permute(1, 0, 2).contiguous()


    # chunk_size = int(y_pred.size()[0] / num_model)
    # y_p = torch.split(y_pred, split_size_or_sections=chunk_size, dim=0)
    # y_p_all = 0
    # for i in range(num_model):
    #     y_p_all += y_p[i]
    # Ensemble = Entropy(y_p_all / num_model)
    # return Ensemble

# def log_det_hp(attention_probs, num_model=args.num_models):
#     b_size = attention_probs[0].size()[0]
#     num_heads = attention_probs[0].size()[1]
#     seq_len = attention_probs[0].size()[2]
#
#     attention_probs = torch.stack(attention_probs)
#
#     a_mask = torch.eye(seq_len, dtype=torch.bool).cuda(args.rank)
#     a_mask=torch.where(a_mask==False,True,False)
#
#     attention_probs=torch.sum(attention_probs,dim=2)
#
#     mask_a_probs = torch.masked_select(attention_probs, a_mask)
#
#     mask_a_probs=mask_a_probs.reshape(num_model,-1, seq_len-1)
#
#     attention_det=mask_a_probs.permute(1,0,2).contiguous() #[da,model_number,seq_len]
#
#
#     attention_det = attention_det / torch.norm(attention_det, dim=-1, keepdim=True)
#
#     matrix = torch.matmul(attention_det, torch.transpose(attention_det, -2, -1))
#     a_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
#
#     #a_log_det = a_log_det.reshape(b_size,num_heads, -1)
#     a_log_det=a_log_det.reshape(b_size,-1)
#
#     #a_log_det = torch.mean(a_log_det, dim=-2)
#
#     a_log_det=torch.mean(a_log_det,dim=-1)
#
#     return a_log_det

def log_det_hp(attention_probs, num_model=args.num_models):
    b_size = attention_probs[0].size()[0]
    num_heads = attention_probs[0].size()[1]
    seq_len = attention_probs[0].size()[2]

    attention_probs = torch.stack(attention_probs)

    #a_mask = torch.eye(seq_len, dtype=torch.bool).cuda(args.rank)
    #a_mask=torch.where(a_mask==False,True,False)

    #attention_probs=torch.sum(attention_probs,dim=2)

    #mask_a_probs = torch.masked_select(attention_probs, a_mask)

    attention_probs=attention_probs.reshape(num_model,-1, seq_len)

    attention_det=attention_probs.permute(1,0,2).contiguous() #[da,model_number,seq_len]


    attention_det = attention_det / torch.norm(attention_det, dim=-1, keepdim=True)

    matrix = torch.matmul(attention_det, torch.transpose(attention_det, -2, -1))
    a_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))

    #a_log_det = a_log_det.reshape(b_size,num_heads, -1)
    a_log_det=a_log_det.reshape(b_size,-1)

    a_log_det = torch.mean(a_log_det, dim=-2)

    a_log_det=torch.mean(a_log_det,dim=-1)

    return a_log_det





def log_det_att(attention_probs, num_model=args.num_models):
    """attention_probs: 是包含num_model个tensor的list，每个tensor [b_size, num_heads, seq_len, seq_len]"""
    # b_size = attention_probs[0].size()[0]
    # num_heads = attention_probs[0].size()[1]
    # seq_len = attention_probs[0].size()[2]
    # attention_probs = torch.cat(attention_probs, dim=0)
    # attention_probs = attention_probs.reshape(num_model, b_size, num_heads, seq_len, seq_len)
    # # print('attention_probs', attention_probs)  # [num_model, b_size, num_heads, seq_len, seq_len]
    # # print('最里维度之和:', torch.sum(attention_probs[0,0,0,0,:])) # 1
    # all_log_det = []
    # for b in range(b_size):  # 对每条数据
    #     # print(b)
    #     a_sent_log_det = 0
    #     att_probs = attention_probs[:, b, :, :, :]  # [num_model, num_heads, seq_len, seq_len] 该数据的attention信息
    #     # print('att_probs', att_probs)
    #     for h in range(num_heads):  # 对于每个attention头
    #         for s in range(seq_len):  # 对于句子中的每个单词
    #             a_probs = att_probs[:, h, s, :]  # [num_model, seq_len]  该头 该句子的该单词 所有模型中其他单词对它的影响
    #             # print('a_probs', a_probs)
    #             # a_probs[:, s] = 0
    #             a_mask = torch.ones_like(a_probs, dtype=torch.bool)
    #             a_mask[:, s] = False  # [num_model, seq_len] s列为False
    #             # print(a_mask.byte())
    #             mask_a_probs = torch.masked_select(a_probs, a_mask)  # [num_model, seq_len-1] 对于a_probs 保留除了s列以外的
    #             # print('mask_a_probs', mask_a_probs)
    #             mask_a_probs = mask_a_probs.reshape(num_model, -1)
    #             # mask_a_probs = mask_a_probs / torch.sum(mask_a_probs, dim=-1, keepdim=True)  # [num_model, seq_len-1] 按行除以行和
    #             mask_a_probs = mask_a_probs / torch.norm(mask_a_probs, dim=-1, keepdim=True)
    #
    #             # print('mask_a_probs', mask_a_probs)
    #             matrix = torch.matmul(mask_a_probs, torch.transpose(mask_a_probs, 1, 0))  # [num_model, num_model]
    #             # print('matrix', matrix)
    #             a_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))  # 数值
    #             a_sent_log_det += a_log_det
    #             # print('all_log_det', all_log_det)
    #     a_sent_log_det = a_sent_log_det / (seq_len * num_heads)
    #     all_log_det.append(a_sent_log_det)
    # all_log_det = torch.cat(all_log_det, dim=0)  # [b_size, 1]
    # # print(all_log_det)
    # # exit(0)

    attention_probs = torch.cat(attention_probs, dim=0)
    b_size = int(attention_probs.size()[0]/num_model)
    attention_probs = attention_probs.reshape(b_size*num_model, -1)

    y_pred_new = []
    for b in range(b_size):
        y_pred_asent = attention_probs[b::b_size]  # num_models, n_labels
        y_pred_new.append(y_pred_asent)
    y_pred_new = torch.cat(y_pred_new, dim=0)
    y_pred_new = y_pred_new.reshape(b_size, args.num_models, -1)
    # print(y_pred_new.size(), y_pred_new)
    matrix = torch.matmul(y_pred_new, torch.transpose(y_pred_new, 2, 1))  # num_models,num_models
    # print(matrix)
    all_log_det = torch.logdet(matrix + det_offset * torch.unsqueeze(torch.eye(num_model).cuda(args.rank), 0))
    # print(all_log_det)
    #     # exit(0)

    return all_log_det





def Loss_withEE_DPP(y_true, y_pred, aux_outs, num_model=args.num_models): # y_pred [batch_size*num_models, num_classes]
    chunk_size = int(y_true.size()[0]/num_model)
    y_p = torch.split(y_pred, split_size_or_sections=chunk_size, dim=0)
    y_t = torch.split(y_true, split_size_or_sections=chunk_size, dim=0)
    CE_all = 0
    for i in range(num_model):
        CE_all += CEloss(y_p[i], y_t[i])
    # if args.lamda == 0 and args.log_det_lamda == 0:
    #     return CE_all
    y_pred = nn.functional.softmax(y_pred, dim=-1)
    y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)  # 截断，防止过小导致后续log操作出现-inf

    b_size = chunk_size
    EE, log_dets, EE_att, log_dets_att = [torch.tensor([0]*b_size, dtype=torch.float).cuda(args.rank) for _ in range(4)]

    if args.lamda > 0:
        EE = Ensemble_Entropy(y_pred)
    if args.log_det_lamda > 0:
        log_dets = log_det(y_pred)

    if args.perturb_attention:
        if args.log_det_att > 0:
            log_dets_att = log_det_hp(aux_outs)
        if args.lamda_att > 0:
            EE_att = Ensemble_Entropy_hp(aux_outs)

    return CE_all - args.lamda * EE - args.log_det_lamda * log_dets - args.log_det_att * log_dets_att - args.lamda_att * EE_att, \
           CE_all, -args.lamda * EE, -args.log_det_lamda * log_dets, -args.log_det_att * log_dets_att, -args.lamda_att * EE_att


def train_epoch(epoch, highest_test_acc, lowest_attention_det, lowest_test_acc, model, optimizer, dataloader_train, dataloader_test, tokenizer):
    """Train for a epoch"""
    time_start = time.time()
    model.train()
    cnt = 0
    all_loss, all_loss_ce, all_loss_e, all_loss_d, all_loss_d_att, all_loss_e_att = [0] * 6
    for idx, (*train_x, train_y) in enumerate(dataloader_train):
        #print("batch:{:d} / {:d}".format(idx,len(dataloader_train)))
        optimizer.zero_grad()
        input_ids, input_mask, segment_ids, train_y = train_x[0].cuda(args.rank), train_x[1].cuda(args.rank), train_x[2].cuda(args.rank), train_y.cuda(args.rank)
        cnt += 1
        aux_outs = None
        if torch.cuda.device_count() > 1:
            model.module.inference = False
            if args.perturb_attention:
                aux_outs, outputs = model.module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            else:
                outputs = model.module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        else:
            model.inference = False
            if args.perturb_attention:
                aux_outs, outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            else:
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        logits = outputs.logits
        true_labels = [train_y for i in range(args.num_models)]
        logits = torch.cat(logits, dim=0)
        true_labels = torch.cat(true_labels, dim=0)

        # loss = criterion(output, train_y)
        loss, loss_ce, loss_e, loss_d, loss_d_att, loss_e_att = Loss_withEE_DPP(true_labels, logits, aux_outs)
        all_loss += loss.mean().item()
        all_loss_ce += loss_ce.mean().item()
        all_loss_e += loss_e.mean().item()
        all_loss_d += loss_d.mean().item()
        all_loss_d_att += loss_d_att.mean().item()
        all_loss_e_att += loss_e_att.mean().item()
        loss = loss.mean()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
    # lr_decay = 0.1
    # if epoch > 50:
    #     if lr_decay > 0:
    #         optimizer.param_groups[0]['lr'] *= lr_decay

    # For process 0
    if is_main_process() and epoch % 1 == 0:
        all_loss = all_loss / len(dataloader_train)
        all_loss_ce = all_loss_ce / len(dataloader_train)
        all_loss_e = all_loss_e / len(dataloader_train)
        all_loss_d = all_loss_d / len(dataloader_train)
        all_loss_d_att = all_loss_d_att / len(dataloader_train)
        all_loss_e_att = all_loss_e_att / len(dataloader_train)

        test_acc = eval_model(model, dataloader_test)
        time_end = time.time()
        time_used = time_end - time_start
        print("Epoch={} time={:.2f}s train_loss={:.6f} loss_ce={:.6f} loss_e={:.6f} loss_d={:.6f} loss_d_att={:.6f} loss_e_att={:.6f} test_acc={:.6f}".format(
                epoch, time_used, all_loss, all_loss_ce, all_loss_e, all_loss_d, all_loss_d_att, all_loss_e_att, test_acc))
        # 保存test acc最高的
        if test_acc > highest_test_acc:
            highest_test_acc = test_acc
            if not os.path.exists(args.save_path): os.makedirs(args.save_path)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.save_path + '/pytorch_model.bin')
                model.module.config.to_json_file(args.save_path + '/config.json')
            else:
                torch.save(model.state_dict(), args.save_path + '/pytorch_model.bin')
                model.config.to_json_file(args.save_path + '/config.json')
            tokenizer.save_vocabulary(args.save_path)
            # print('save model when test acc=', test_acc)
        # 保存attention det最低的
        if all_loss_d_att < lowest_attention_det:
            lowest_attention_det = all_loss_d_att
            save_path = args.save_path + '_lowest_att_det'
            if not os.path.exists(save_path): os.makedirs(save_path)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path + '/pytorch_model.bin')
                model.module.config.to_json_file(save_path + '/config.json')
            else:
                torch.save(model.state_dict(), save_path + '/pytorch_model.bin')
                model.config.to_json_file(save_path + '/config.json')
            tokenizer.save_vocabulary(save_path)
        # 保存第10个epoch之后，test acc最低的
        if epoch > 10 and test_acc < lowest_test_acc:
            lowest_test_acc = test_acc
            save_path = args.save_path + '_lowest_acc'
            if not os.path.exists(save_path): os.makedirs(save_path)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path + '/pytorch_model.bin')
                model.module.config.to_json_file(save_path + '/config.json')
            else:
                torch.save(model.state_dict(), save_path + '/pytorch_model.bin')
                model.config.to_json_file(save_path + '/config.json')
            tokenizer.save_vocabulary(save_path)

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
    return highest_test_acc, lowest_attention_det, lowest_test_acc


def main():

    """Setting for parallel training"""
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # args.rank = 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        # test_data_list, test_label_list = test_data_list[:2], test_label_list[:2]

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
    config = BertConfig.from_pretrained(args.target_model_path)
    config.aux_weight = args.aux_weight
    config.num_models = args.num_models
    config.perturb_attention = args.perturb_attention
    checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')

    if 'esb' in args.target_model_path:
        # 从之前工程训练的esb模型中加载
        new_checkpoint = {}
        for name, param in checkpoint.items():
            name_ = name.replace('bert.bert', 'bert').replace('bert.classifier', 'classifier')
            new_checkpoint[name_] = param
        if args.target_model == 'bert':
            model = EnsembleBERT.from_pretrained('bert-base-uncased', state_dict=new_checkpoint, config=config).cuda(args.rank)
        elif args.target_model == 'roberta':
            model = EnsembleBERT.from_pretrained('robert-base', state_dict=new_checkpoint, config=config).cuda(args.rank)
    else:
        if args.target_model == 'bert':
            model = EnsembleBERT.from_pretrained('bert-base-uncased', state_dict=checkpoint, config=config).cuda(args.rank)
        elif args.target_model == 'roberta':
            model = EnsembleROBERTA.from_pretrained('bert-base-uncased', state_dict=checkpoint, config=config).cuda(args.rank)

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
    # 关闭bert部分的参数更新，只更新aux部分参数
    if args.target_model == 'bert':
        if torch.cuda.device_count() > 1:
            model.module.requires_grad_(False)
            if args.aux_weight > 0:
                model.module.bert.encoder.aux.requires_grad_(True)
            if args.perturb_attention:
                model.module.bert.encoder.layer[args.modif_att_layer].attention.self.aux_qs.requires_grad_(True)
                model.module.bert.encoder.layer[args.modif_att_layer].attention.self.aux_ks.requires_grad_(True)
                # model.module.bert.encoder.layer[args.modif_att_layer].attention.self.aux_vs.requires_grad_(True)
        else:
            model.requires_grad_(False)
            if args.aux_weight > 0:
                model.bert.encoder.aux.requires_grad_(True)
            if args.perturb_attention:
                model.bert.encoder.layer[args.modif_att_layer].attention.self.aux_qs.requires_grad_(True)
                model.bert.encoder.layer[args.modif_att_layer].attention.self.aux_ks.requires_grad_(True)
                # model.bert.encoder.layer[args.modif_att_layer].attention.self.aux_vs.requires_grad_(True)
    elif args.target_model == 'roberta':
        if torch.cuda.device_count() > 1:
            model.module.requires_grad_(False)
            if args.aux_weight > 0:
                model.module.roberta.encoder.aux.requires_grad_(True)
            if args.perturb_attention:
                model.module.roberta.encoder.layer[args.modif_att_layer].attention.self.aux_qs.requires_grad_(True)
                model.module.roberta.encoder.layer[args.modif_att_layer].attention.self.aux_ks.requires_grad_(True)
                # model.module.roberta.encoder.layer[args.modif_att_layer].attention.self.aux_vs.requires_grad_(True)
        else:
            model.requires_grad_(False)
            if args.aux_weight > 0:
                model.roberta.encoder.aux.requires_grad_(True)
            if args.perturb_attention:
                model.roberta.encoder.layer[args.modif_att_layer].attention.self.aux_qs.requires_grad_(True)
                model.roberta.encoder.layer[args.modif_att_layer].attention.self.aux_ks.requires_grad_(True)
                # model.roberta.encoder.layer[args.modif_att_layer].attention.self.aux_vs.requires_grad_(True)

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
    highest_test_acc = 0
    lowest_attention_det = 1000  # attention行列式
    lowest_test_acc = 100
    for e in range(epoch):
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

        if torch.cuda.device_count() > 1:
            train_sampler.set_epoch(epoch)
        highest_test_acc, best_attention_det, worst_test_acc = train_epoch(e, highest_test_acc, lowest_attention_det, lowest_test_acc, model, optimizer, dataloader_train, dataloader_test, tokenizer)


if __name__ == '__main__':
    main()




