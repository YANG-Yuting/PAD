import datasets
import numpy as np
import torch
import os, time
import torch.nn as nn
import copy
import seaborn as sns
import matplotlib.pyplot as plt

"""from mine"""
from data_utils import read_text, dataset_mapping, dataset_mapping_trec, MyDataset
import transformers
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification
from multi_train_utils.distributed_utils import init_distributed_mode, dist, is_main_process
from models import EnsembleBERT
from config import args
from models import eval_model

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rc('font', family='Times New Roman')

    font = {'size': 14}
    plt.figure(figsize=(21, 4))

    x = ['t1', 't2', 't3', 't4']
    k1 = [0.4302, 0.3571, 0.3743, 0.3464]
    plt.subplot(1, 4, 1)
    plt.plot(x, k1, '+-', color='#8E8BFE', label="DeepWordBug", ms=5)  # s-:方形
    plt.scatter('t1', 0.4302, marker='^')
    # plt.axhline(y=0.4835,ls='--', c='blue')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("Trigger", font)
    plt.ylabel("Succ", font)

    x = ['random', 'importance']
    k1 = [0.4302, 0.2500]
    plt.subplot(1, 4, 2)
    plt.plot(x, k1, '+-', color='#8E8BFE', label="DeepWordBug", ms=5)  # s-:方形
    plt.scatter('random', 0.4302, marker='^')
    # plt.axhline(y=0.4835,ls='--', c='blue')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("Mask strategy", font)

    x = ['10', '50', '100']
    k1 = [0.2812, 0.4302, 0.5208]
    plt.subplot(1, 4, 3)
    plt.plot(x, k1, '+-', color='#8E8BFE', label="DeepWordBug", ms=5)  # s-:方形
    plt.scatter('50', 0.4302, marker='^')
    # plt.axhline(y=0.4835,ls='--', c='blue')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("Sample size", font)

    x = ['0.15', '0.25', '0.5']
    k1 = [0.4302, 0.5521, 0.8333]
    plt.subplot(1, 4, 4)
    plt.plot(x, k1, '+-', color='#8E8BFE', label="DeepWordBug", ms=5)  # s-:方形
    plt.scatter('0.15', 0.4302, marker='^')
    # plt.axhline(y=0.4835,ls='--', c='blue')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("Mask ratio", font)

    plt.savefig('./figs/prompt_design.png', dpi=600, bbox_inches='tight')
    plt.show()

    exit(0)
    args.rank = 0
    torch.cuda.set_device(args.rank)
    device = torch.device('cuda', args.rank)

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
        test_data_list, test_label_list = test_data_list[:2], test_label_list[:2]

    """Initialize model"""
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    config = BertConfig.from_pretrained(args.target_model_path)
    config.aux_weight = args.aux_weight
    config.num_models = args.num_models
    checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')

    if 'esb' in args.target_model_path:
        # 从之前工程训练的esb模型中加载
        new_checkpoint = {}
        for name, param in checkpoint.items():
            name_ = name.replace('bert.bert', 'bert').replace('bert.classifier', 'classifier')
            new_checkpoint[name_] = param
        model = EnsembleBERT.from_pretrained('bert-base-uncased', state_dict=new_checkpoint, config=config).cuda(args.rank)
    else:
        model = EnsembleBERT.from_pretrained('bert-base-uncased', state_dict=checkpoint, config=config).cuda(args.rank)

    model.eval()

    # build dataloader
    dataset_test, _ = MyDataset(args, tokenizer).transform_text(test_data_list, test_label_list)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, sampler=test_sampler, batch_size=args.batch_size)

    data_no = 0  # 观察哪条数据
    head_no = 0  # 观察哪个头
    save_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/figs/heatmaps/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    with torch.no_grad():
        for idx, (*test_x, test_y) in enumerate(dataloader_test):
            input_ids, input_mask, segment_ids, train_y = test_x[0].cuda(args.rank), test_x[1].cuda(args.rank), test_x[2].cuda(args.rank), test_y.cuda(args.rank)
            data_len = np.where(input_ids[data_no].cpu().numpy() == tokenizer.pad_token_id)[0][0]
            print(data_len)
            model.inference = False
            _, outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_attentions=True)
            all_attentions = outputs.attentions
            all_logits = outputs.logits

            # 观察第0层
            attentions = []
            for n in range(args.num_models):
                attentions.append(all_attentions[n][head_no])
            attentions = torch.stack(attentions)  # [num_models, batch_size, num_heads, seq_len, seq_len]
            logits = torch.stack(all_logits)  # [num_models, batch_size, num_labels]

            # 观察第0条数据的第0个头
            attentions = attentions[:, data_no, head_no, :, :]  # [num_models, seq_len, seq_len]

            # 观察每个词对于其他词语的影响
            all_importances = []
            print(train_y[data_no], tokenizer.convert_ids_to_tokens(input_ids[data_no].cpu().numpy().tolist()))
            for n in range(args.num_models):
                atts = attentions[n, :data_len, :data_len]  # [seq_len, seq_len]
                all_importances.append(atts)

                # 对于cls位置的影响
                # importances = atts[0, :]  # cls位置
                # values, indices = torch.topk(importances, k=20, dim=-1)
                # topk_ids = input_ids[0].index_select(dim=0, index=indices)

                # 对于所有词的影响
                # importances = torch.sum(atts, dim=0).reshape(1, -1)  # 按列求和
                # values, indices = torch.topk(importances, k=20, dim=-1)
                # topk_ids = input_ids[0].index_select(dim=0, index=indices[0])
                #
                # print(n, tokenizer.convert_ids_to_tokens(topk_ids.cpu().numpy().tolist()))
                # print(logits[n, 0, :])

            f, (ax1, ax2, ax3) = plt.subplots(figsize=(18,6), nrows=1, ncols=3)
            sns.heatmap(data=all_importances[0].cpu().numpy(), ax=ax1, vmax=1, vmin=0)
            sns.heatmap(data=all_importances[1].cpu().numpy(), ax=ax2, vmax=1, vmin=0)
            sns.heatmap(data=all_importances[2].cpu().numpy(), ax=ax3, vmax=1, vmin=0)
            scatter_fig = f.get_figure()
            scatter_fig.savefig(save_path + '%s_%s.png' % (args.task, args.target_model), dpi=400)

            # print(len(all_attentions))
            # print(len(all_attentions[0]))
            # print(all_attentions[0][0].size())
            # all_attentions = torch.stack(all_attentions)
            # print(all_attentions.size())


            # # 12 torch.Size([1, 12, 256, 256]) batch_size, num_head, input_len, input_len
            # print(logits)
            # print(len(all_attentions), all_attentions[0].size())
            # # 观察第0层每个位置对于第1层中所有词语的权重的均值
            # all_attentions = all_attentions[0]  # 观察第0层 [1, 12, 256, 256]
            # all_attentions = all_attentions[:, 0, :, :]  # 观察第0个head [1, 256, 256]
            # # temp = all_attentions[:, 0, :]  # sum=1 表示第0个单词的embedding计算时，所有单词的权重
            # out = torch.sum(all_attentions, dim=1)
            # # out_ = torch.sum(all_attentions, dim=2)  # 全为1
            # print(i, out)
            # values, indices = torch.topk(out, k=20, dim=-1)
            # print(values, indices)
            # topk_ids = input_ids[0].index_select(dim=0, index=indices[0])
            # print(model.dataset.tokenizer.convert_ids_to_tokens(topk_ids.cpu().numpy().tolist()))
            #
            # exit(0)





