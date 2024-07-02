import os, re, torch
import numpy as np
import pickle
import copy
import random
from config import args
import OpenAttack_.OpenAttack as oa

def read_text(path, task):
    if task == 'imdb':
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        # train
        train_path = path + '/train'
        train_pos_path = train_path + '/pos'
        train_neg_path = train_path + '/neg'
        train_pos_files = [train_pos_path + '/' + x for x in os.listdir(train_pos_path) if x.endswith('.txt')]
        train_neg_files = [train_neg_path + '/' + x for x in os.listdir(train_neg_path) if x.endswith('.txt')]

        # train_pos_list = [open(x, 'r').read().lower() for x in train_pos_files]
        # train_neg_list = [open(x, 'r').read().lower() for x in train_neg_files]
        # tiz for IMDB dataset with punctuation
        train_pos_list, train_neg_list = [], []
        for x in train_pos_files:
            t1 = open(x, 'r').read().lower().replace('<br /><br />',' ').replace('\x85', '')
            t2 = re.split(r"([.。!！?？；;，,+])", t1)
            t3 = ' '.join(t2)
            t3 = re.sub(' +', ' ', t3)
            if t3[-1] == ' ':
                t3 = t3[:-1]
            t3 = t3.split(' ')
            train_pos_list.append(t3)
        for x in train_neg_files:
            t1 = open(x, 'r').read().lower().replace('<br /><br />', ' ').replace('\x85', '')
            t2 = re.split(r"([.。!！?？；;，,+])", t1)
            t3 = ' '.join(t2)
            t3 = re.sub(' +', ' ', t3)
            if t3[-1] == ' ':
                t3 = t3[:-1]
            t3 = t3.split(' ')
            train_neg_list.append(t3)

        # train_pos_list = [' '.join(re.split(r"([.。!！?？；;，,+])", open(x, 'r').read().lower().replace('<br /><br />',' ').replace('  ',' ').replace('  ',' ').replace('  ',' '))) for x in train_pos_files]  #tiz'211222
        # train_neg_list = [' '.join(re.split(r"([.。!！?？；;，,+])", open(x, 'r').read().lower().replace('<br /><br />',' ').replace('  ',' ').replace('  ',' ').replace('  ',' '))) for x in train_neg_files]  #tiz'211222


        # 20220805 特殊符号删除 保存至imdb/dataset_50000_has_punctuation_.pkl
        # filters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n\x85'  # 句子符号不删：!.,?
        # filters = '\x85'
        # for f in filters:
        #     train_pos_list = [t.replace(f, '') for t in train_pos_list]
        #     train_neg_list = [t.replace(f, '') for t in train_neg_list]
        train_data_list = train_pos_list + train_neg_list
        train_label_list = [1] * len(train_pos_list) + [0] * len(train_neg_list)

        # test
        test_path = path + '/test'
        test_pos_path = test_path + '/pos'
        test_neg_path = test_path + '/neg'
        test_pos_files = [test_pos_path + '/' + x for x in os.listdir(test_pos_path) if x.endswith('.txt')]
        test_neg_files = [test_neg_path + '/' + x for x in os.listdir(test_neg_path) if x.endswith('.txt')]

        # test_pos_list = [open(x, 'r').read().lower() for x in test_pos_files]
        # test_neg_list = [open(x, 'r').read().lower() for x in test_neg_files]
        # tiz for IMDB dataset with punctuation

        # test_pos_list = [' '.join(re.split(r"([.。!！?？；;，,+])", open(x, 'r').read().lower().replace('<br /><br />',' '))).replace('\x85','').replace('  ', ' ') for x in test_pos_files]  #tiz'211222
        # test_neg_list = [' '.join(re.split(r"([.。!！?？；;，,+])", open(x, 'r').read().lower().replace('<br /><br />',' '))).replace('\x85','').replace('  ', ' ') for x in test_neg_files]  #tiz'211222

        test_pos_list, test_neg_list = [], []
        for x in test_pos_files:
            t1 = open(x, 'r').read().lower().replace('<br /><br />', ' ').replace('\x85', '')
            t2 = re.split(r"([.。!！?？；;，,+])", t1)
            # t3 = ' '.join(t2)
            # t3 = re.sub(' +', ' ', t3)
            # if t3[-1] == ' ':
            #     t3 = t3[:-1]
            test_pos_list.append(t2)
        for x in test_neg_files:
            t1 = open(x, 'r').read().lower().replace('<br /><br />', ' ').replace('\x85', '')
            t2 = re.split(r"([.。!！?？；;，,+])", t1)
            # t3 = ' '.join(t2)
            # t3 = re.sub(' +', ' ', t3)
            # if t3[-1] == ' ':
            #     t3 = t3[:-1]
            test_neg_list.append(t2)

        test_data_list = test_pos_list + test_neg_list
        test_label_list = [1] * len(test_pos_list) + [0] * len(test_neg_list)

        # imdb打乱训练更好
        if args.task == 'imdb':
            c = list(zip(train_data_list, train_label_list))
            random.shuffle(c)
            train_data_list, train_label_list = zip(*c)

    elif task == 'mr':
        train_data_list = []
        train_label_list = []
        train_lines = open(path + '/train.txt', 'r').read().lower().splitlines()
        for line in train_lines:
            train_data_list.append(line.split(' ')[1:])
            train_label_list.append(int(line.split(' ')[0]))

        test_data_list = []
        test_label_list = []
        test_lines = open(path + '/test.txt', 'r').read().lower().splitlines()
        for line in test_lines:
            test_data_list.append(line.split(' ')[1:])
            test_label_list.append(int(line.split(' ')[0]))
    elif task == 'snli':
        with open(path + '/nli_tokenizer.pkl', 'rb') as fh:
            snli_tokenizer = pickle.load(fh)
        with open(path + '/all_seqs.pkl', 'rb') as fh:
            train, _, test = pickle.load(fh)
        full_dict = {w: i for (w, i) in snli_tokenizer.word_index.items()}
        inv_full_dict = {i: w for (w, i) in full_dict.items()}
        full_dict['<oov>'] = len(full_dict)  # 42391
        inv_full_dict[len(full_dict) - 1] = '<oov>'
        full_dict['<pad>'] = len(full_dict)  # 42391
        inv_full_dict[len(full_dict) - 1] = '<pad>'

        null_idx = [i for i in range(len(test['s2'])) if len(test['s2'][i]) <= 2]
        test['s1'] = np.delete(test['s1'], null_idx)
        test['s2'] = np.delete(test['s2'], null_idx)
        test['label'] = np.delete(test['label'], null_idx)
        test_s1 = [[inv_full_dict[w] for w in t[1:-1]] for t in test['s1']]
        test_s2 = [[inv_full_dict[w] for w in t[1:-1]] for t in test['s2']]

        null_idx = [i for i in range(len(train['s2'])) if len(train['s2'][i]) <= 2]
        train['s1'] = np.delete(train['s1'], null_idx)
        train['s2'] = np.delete(train['s2'], null_idx)
        train['label'] = np.delete(train['label'], null_idx)
        train_s1 = [[inv_full_dict[w] for w in t[1:-1]] for t in train['s1']]
        train_s2 = [[inv_full_dict[w] for w in t[1:-1]] for t in train['s2']]

        test_label_list = list(test['label'])
        train_label_list = list(train['label'])
        train_data_list, test_data_list, = [], []
        for s1, s2 in zip(test_s1, test_s2):
            test_data_list.append(s1 + [args.sep_token] + s2)
        for s1, s2 in zip(train_s1, train_s2):
            train_data_list.append(s1 + [args.sep_token] + s2)

    # a = list(zip(train_data_list, train_label_list))
    # # random.shuffle(a) # 20220106为了保持imdb有无符号的dataset包含相同顺序的数据，不打乱
    # train_data_list, train_label_list = zip(*a)
    #
    # b = list(zip(test_data_list, test_label_list))
    # # random.shuffle(b)
    # test_data_list, test_label_list = zip(*b)

    return train_data_list, train_label_list, test_data_list, test_label_list


def dataset_mapping(data_list, label_list):
    data = []
    for text, label in zip(data_list, label_list):
        if isinstance(text, list):
            text = ' '.join(text)
        data.append({'x': text, 'y': label})
    return data


label2id_model = {"ABBR": 2, "DESC": 0, "ENTY": 1, "HUM": 3, "LOC": 5, "NUM": 4}  # from trec-coarse/config.json
id2label_dataset = {0: "ABBR", 2: "DESC", 1: "ENTY", 3: "HUM", 4: "LOC", 5: "NUM"}
def dataset_mapping_trec(x):
    label = label2id_model[id2label_dataset[x["coarse_label"]]]  # 注意：模型的id-label对应关系与数据集中的不一致
    return {
        "x": x["text"],
        "y": label,
    }

# imdb数据集！你怎么这么奇怪！！！
special_symbols = ['\'', '.', '-', '`', '/', '(', ')', '"', ':', '*', '´', '<UNK>', ' ', '&', '\t', '$', '<', '>', '%',
                   '@', '«', '»', '#', '[', ']', '¨¦', '\x91', '£', '{', '}', '~', '\x97', '_', '|', '¤', '\xa0', '’',
                   '¡¦', '^', '¨', '=', '\\', '¡', '₤', '\x96', '§', '¿', '®', '\x80', '‘', '\x84', '\x08', '“', '”',
                   '\x9e', '\x8e', '\xad', '‘', '…', '\x9a', ',', '!', ';', '+', '?']
def get_syn(examples, labels):
    punctTokenizer = oa.text_process.tokenizer.punct_tokenizer.PunctTokenizer()  # 词性标注
    wordSubstitute = oa.attack_assist.substitute.word.english_wordnet.WordNetSubstitute(k=args.syn_num)  # 获得同义词，保留top5

    candidate_sents_all, candidate_labels_all, syn_flag_all = [], [], []
    for sent, label in zip(examples, labels):  # inputs: CLS text_a SEP
        if isinstance(sent, str):
            sent = sent.split(' ')
        if len(sent) > (args.max_seq_length - 2):
            sent = sent[:(args.max_seq_length - 2)]
        candidate_labels = [label] * args.syn_num
        candidate_sents = np.array([sent] * args.syn_num)
        syn_flag = np.array([[0] * args.max_seq_length] * args.syn_num)

        sent_4token = []
        for wt in sent:
            for ss in special_symbols:
                if ss in wt:  # wt若包含上述字符串，词性标注后会被拆分成多个词，导致tokens和sent长度不一致
                    wt = wt.replace(ss, '')
            if len(wt) == 0:  # 若wt为空，替换成'a'之后，pos为'other'，则不会获得同义词，则该位置仍然是原本的词
                wt = 'a'
            sent_4token.append(wt)
        tokens = punctTokenizer.tokenize(' '.join(sent_4token))
        # if len(sent_4token) != len(tokens):
        #     print(sent_4token)
        #     print(tokens)
        # assert len(sent) == len(sent_4token)  # 词性标注前后句子长度可能会不一样
        # assert len(sent) == len(tokens)  # 词性标注前后句子长度可能会不一样
        for idx, token in enumerate(tokens):
            word, pos = token
            # if word != sent_4token[idx]:
            #     print('------')
            #     print(idx, word, sent_4token[idx])
            #     print(sent, sent_4token)
            #     break

            if pos != 'other':
                syns = wordSubstitute.substitute(word, pos)
                if len(syns) > 0:
                    syns = copy.deepcopy([syn[0] for syn in syns])
                    try:
                        candidate_sents[:len(syns), idx] = copy.deepcopy(syns)
                        syn_flag[:len(syns), idx+1] = 1  # 第0个位置留出给cls
                    except:
                        print(len(sent_4token), sent_4token)
                        print(len(tokens), tokens)
                        print(word, sent_4token[idx])
                        exit(0)

        candidate_sents_all.extend(candidate_sents.tolist())
        candidate_labels_all.extend(candidate_labels)
        syn_flag_all.extend(syn_flag.tolist())


    return candidate_sents_all, candidate_labels_all, syn_flag_all
    # return syn_ids_all, syn_input_mask_all, syn_segment_ids_all, syn_flag_all, syn_labels_all


class MyDataset(torch.utils.data.Dataset):
    """ Dataset for bert/rpberta
        Mind: for snli, s2 is concatenated with s1 using corresponding sep_token of tokenizer
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def convert_examples_to_features(self, examples, labels):
        if isinstance(examples[0], list):
            data = []
            for d in examples:
                d = ' '.join(d)
                data.append(d)
            examples = copy.deepcopy(data)
        outs = self.tokenizer(examples, truncation=True, max_length=args.max_seq_length, padding='max_length')
        input_ids_all = outs['input_ids']
        input_mask_all = outs['attention_mask']
        if args.target_model == 'bert':
            segment_ids_all = outs['token_type_ids']
        else:  # robertatokenizer无token_type_ids返回值
            segment_ids_all = [[0]*args.max_seq_length for _ in range(len(examples))]
        labels_all = labels
        # print(input_ids_all[:2])
        # print(segment_ids_all[:2])
        # print(input_mask_all[:2])
        # exit(0)

        # input_ids_all, input_mask_all, segment_ids_all, labels_all = [], [], [], []
        # for (ex_index, text_a) in enumerate(examples):  # inputs: CLS text_a SEP
        #     # If SNLI, text_a should be: s1 "SEP" s2
        #     if isinstance(text_a, list):
        #         text_a = ' '.join(text_a)
        #     tokens_a = self.tokenizer.tokenize(text_a)  # 109
        #
        #     # Account for [CLS] and [SEP] with "- 2"
        #     if len(tokens_a) > self.args.max_seq_length - 2:
        #         tokens_a = tokens_a[:(self.args.max_seq_length - 2)]
        #
        #     tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        #     segment_ids = [0] * len(tokens)
        #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #
        #     # The mask has 1 for real tokens and 0 for padding tokens. Only real
        #     # tokens are attended to.
        #     input_mask = [1] * len(input_ids)
        #
        #     # Zero-pad up to the sequence length.
        #     padding = [0] * (self.args.max_seq_length - len(input_ids))
        #     input_ids += padding
        #     input_mask += padding
        #     segment_ids += padding
        #
        #     input_ids_all.append(input_ids)
        #     input_mask_all.append(input_mask)
        #     segment_ids_all.append(segment_ids)
        #     labels_all.append(labels[ex_index])
        return input_ids_all, input_mask_all, segment_ids_all, labels_all

    def transform_text(self, data, labels):
        input_ids_all, input_mask_all, segment_ids_all, labels_all = self.convert_examples_to_features(data, labels)

        # if args.num_models > 1:
        #     input_ids_all = torch.unsqueeze(input_ids_all, 0).repeat(args.num_models, 1, 1).view(-1, args.max_seq_length)
        #     input_mask_all = torch.unsqueeze(input_mask_all, 0).repeat(args.num_models, 1, 1).view(-1, args.max_seq_length)
        #     segment_ids_all = torch.unsqueeze(segment_ids_all, 0).repeat(args.num_models, 1, 1).view(-1, args.max_seq_length)
        #     labels_all = torch.unsqueeze(labels_all, 0).repeat(args.num_models, 1).view(-1, 1)
        #     # model_ids_all = []
        #     # for i in range(args.num_models):
        #     #     model_ids_all.append(torch.zeros_like(labels)+i)
        #     # model_ids_all = torch.cat(model_ids_all, dim=0)
        #
        #     all_input_ids = torch.tensor(input_ids_all, dtype=torch.long)
        #     all_input_mask = torch.tensor(input_mask_all, dtype=torch.long)
        #     all_segment_ids = torch.tensor(segment_ids_all, dtype=torch.long)
        #     all_labels = torch.tensor(labels_all, dtype=torch.long)
        #     # all_model_ids = torch.tensor(model_ids_all, dtype=torch.long)
        #     all_data = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
        # else:
        all_input_ids = torch.tensor(input_ids_all, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask_all, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids_all, dtype=torch.long)
        all_labels = torch.tensor(labels_all, dtype=torch.long)
        all_data = torch.utils.data.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

        if self.args.ascc:
            # get 1-hop syns
            candidate_sents_all, candidate_labels_all, syn_flag_all = get_syn(data, labels)
            syn_ids_all, syn_input_mask_all, syn_segment_ids_all, syn_labels_all = self.convert_examples_to_features(candidate_sents_all, candidate_labels_all)

            syn_ids_all = torch.tensor(syn_ids_all, dtype=torch.long)
            syn_input_mask_all = torch.tensor(syn_input_mask_all, dtype=torch.long)
            syn_segment_ids_all = torch.tensor(syn_segment_ids_all, dtype=torch.long)
            syn_flag_all = torch.tensor(syn_flag_all, dtype=torch.long)
            syn_labels_all = torch.tensor(syn_labels_all, dtype=torch.long)
            all_data_syn = torch.utils.data.TensorDataset(syn_ids_all, syn_input_mask_all, syn_segment_ids_all, syn_flag_all, syn_labels_all)
            assert len(syn_ids_all) == self.args.syn_num * len(input_ids_all)
            return all_data, all_data_syn
        elif self.args.dne:
            # get 2-hop syns
            candidate_sents_all_1hop, candidate_labels_all_1hop, syn_flag_all_1hop = get_syn(data, labels)
            candidate_sents_all_2hop, candidate_labels_all_2hop, syn_flag_all_2hop = get_syn(candidate_sents_all_1hop, candidate_labels_all_1hop)
            syn_ids_all, syn_input_mask_all, syn_segment_ids_all, syn_labels_all = self.convert_examples_to_features(candidate_sents_all_2hop, candidate_labels_all_2hop)

            syn_ids_all = torch.tensor(syn_ids_all, dtype=torch.long)
            syn_input_mask_all = torch.tensor(syn_input_mask_all, dtype=torch.long)
            syn_segment_ids_all = torch.tensor(syn_segment_ids_all, dtype=torch.long)
            syn_flag_all = torch.tensor(syn_flag_all_2hop, dtype=torch.long)
            syn_labels_all = torch.tensor(syn_labels_all, dtype=torch.long)
            all_data_syn = torch.utils.data.TensorDataset(syn_ids_all, syn_input_mask_all, syn_segment_ids_all, syn_flag_all, syn_labels_all)
            assert len(syn_ids_all) == self.args.syn_num * self.args.syn_num * len(input_ids_all)
            return all_data, all_data_syn

        else:
            return all_data, None
