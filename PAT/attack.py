import sys
sys.path.append('..')
import OpenAttack_.OpenAttack as oa
import datasets
import numpy as np
import torch, os
import torch.nn.functional as F
import json, math, pickle, itertools, nltk,re

"""from mine"""
from data_utils import read_text, dataset_mapping, dataset_mapping_trec, get_syn, MyDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, RobertaTokenizer, BertTokenizer, GPT2Tokenizer, BartTokenizer, GPT2LMHeadModel, pipeline, BertForMaskedLM, RobertaForMaskedLM, BartForConditionalGeneration
from config import args
from models import EnsembleBERT, EnsembleROBERTA

stop_mark = ['.', ',', '?', ';', '!']
model_types = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 'bart': 'facebook/bart-base'}
model_type = model_types[args.target_model]
# prompt_templates = [[], []]
prompt_templates = [['it', 'is', 'a', 'good', 'movie', '.'], ['it', 'is', 'a', 'bad', 'movie', '.']]
# prompt_templates = [['i', 'like', 'the', 'movie', 'so', 'much', '.'], ['i', 'hate', 'the', 'movie', 'so', 'much', '.']]
# prompt_templates = [['it', 'is', 'a', 'funny', 'movie', '.'], ['it', 'is', 'a', 'boring', 'movie', '.']]
# prompt_templates = [['i', 'think', 'it', 'is', 'funny' , '.'], ['i', 'think', 'it', 'is', 'boring','.']]

args.mask_mode = 'selectedPS'
args.sample_size = 50
args.mask_ratio = 0.15
args.topk = 1
args.word_emb = False

"""利用GPT2计算文本困惑度"""
def cal_ppl_bygpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

    """计算GPT2对于一条文本的困惑度"""
    def score(text):
        text = text.replace(' ##','')
        tokenize_input = tokenizer.tokenize(text)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
        loss = model(tensor_input, labels=tensor_input)['loss']
        ppl = math.exp(loss)  # 这是真正的困惑度，返回的是log ppl
        return ppl

    """1. 读取生成的对抗样本"""
    # 确定读取的文件名

    """1.1 prompt产生的"""
    # lpt = True if args.load_prompt_trained else False
    # we = True if args.word_emb else False
    # fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/pat_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead_p0.json' %
    #           (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
    #            args.topk, we, lpt), 'r')
    # # """textfooler产生的"""
    # # fr = open('/pub/data/huangpei/TextFooler/adv_results/test_set/org/tf_%s_%s_adv_symTrue.json' % (args.task, args.target_model), 'r')
    # data = json.load(fr)
    # print('Data size for evaluating:', len(data))
    # """对每条文本进行困惑度评分"""
    # adv_num = 0  # 保存所有的对抗样本个数（每条测试样例可能有多条对抗样本）
    # adv_ppl_sum = 0.0  # 求和所有对抗样本的ppl
    # # 每条样例有多条对抗样本，怎么处理？
    # all_best_ppls = []  # 保存每条测试样例对抗样本中ppl最小的那个 的ppl
    # for idx, dd in data.items():  # 对于每条样例
    #     # start_time = time.clock()
    #     adv_texts = dd['adv_texts']
    #     adv_ppls = []
    #     for adv_text in adv_texts:
    #         adv_text = adv_text[:512]  # gpt的输入要小于512
    #         score_a = score(adv_text)
    #         adv_ppls.append(score_a)
    #         adv_num += 1
    #         adv_ppl_sum += score_a
    #     best_ppl = np.min(np.array(adv_ppls))
    #     all_best_ppls.append(best_ppl)
    #     # data[idx]['adv_ppls'] = adv_ppls
    #     # print('Time for a data:', time.clock()-start_time)

    """1.2 sempso产生的(adv只有一条)"""
    fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/sempso/test_set/org/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'rb')
    # fr = open('/pub/data/huangpei/PAT-AAAI23/TextFooler/adv_exps/sempso/test_set/org/sem_%s_%s_success.pkl' % (args.task, args.target_model), 'rb')
    input_list, true_label_list, output_list, success, change_list, num_change_list, success_time = pickle.load(fr)
    data = output_list
    print('Data size for evaluating:', len(data))
    """对每条文本进行困惑度评分"""
    adv_num = 0  # 保存所有的对抗样本个数（每条测试样例可能有多条对抗样本）
    all_best_ppls = []
    for idx, dd in enumerate(data):
        adv_text = ' '.join(dd)
        adv_text = adv_text[:512]  # gpt的输入要小于512

        score_a = score(adv_text)
        adv_num += 1
        all_best_ppls.append(score_a)


    """3. 保存评分结果"""
    # if args.word_emb:
    #     fw = open('/pub/data/huangpei/TextFooler/prompt_results/adv_ppl_%s_%s_%s_%.2f_%d_%d_we_lpt%s.json' % (
    #     args.task, args.target_model, args.mask_mode, args.mask_ratio, args.sample_size, args.topk, lpt), 'w')
    # else:
    #     fw = open('/pub/data/huangpei/TextFooler/prompt_results/adv_ppl_%s_%s_%s_%.2f_%d_%d_lpt%s.json' % (
    #     args.task, args.target_model, args.mask_mode, args.mask_ratio, args.sample_size, args.topk, lpt), 'w')
    # json.dump(data, fw, indent=4)

    # adv_ppl_mean = adv_ppl_sum/adv_num
    adv_ppl_mean = np.mean(np.array(all_best_ppls))
    print('%d instances, %d adversarial examples, %f mean ppl for GPT2' % (len(data), adv_num, adv_ppl_mean))


"""利用GPT进行续写"""
def gpt_generate(data, args):
    generator = pipeline('text-generation', model='gpt2')
    if args.task == 'imdb':
        output_length = 350
    else:
        output_length = args.max_seq_length + 10
    num_candi_sents = 5
    texts = []
    outs = {}
    for idx, (text, label) in enumerate(data):
        # 文本截断到max_seq_length之内
        text = text[:args.max_seq_length - 9]  # 至少留10个位置用来续写
        # 若文本末尾本身有符号，去掉
        if text[-1] in stop_mark:
            text = text[:-1]
        # 文本末尾加入逗号
        text += [',']
        texts.append(' '.join(text))
        outs[idx] = {'label': label}

    """生成策略"""
    # greedy
    # gpt_output = generator(texts, max_length=output_length, num_return_sequences=num_candi_sents)
    # beam search
    gpt_output = generator(texts, max_length=output_length, num_beams=5, early_stopping=True, num_return_sequences=num_candi_sents, )

    for idx in range(len(gpt_output)):
        candi_sents = [go['generated_text'] for go in gpt_output[idx]]
        outs[idx]['candi_texts'] = candi_sents

    with open('/pub/data/huangpei/TextFooler/prompt_results/gpt12_beamSearch_%s_comma.json' % (args.task), 'w') as fw:
        json.dump(outs, fw, indent=4)

    # 写到第一个句号就结束（减少生成时间）


"""加载攻击模型"""
def build_victim():
    """tokenizer"""
    if args.target_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.target_model_path)
    elif args.target_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.target_model_path)
    elif args.target_model == 'bart':
        tokenizer = BartTokenizer.from_pretrained(args.target_model_path)

    """config & target model"""

    model = AutoModelForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels)
    model.eval()
    if args.target_model == 'bert':
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings, batch_size=args.batch_size)
    elif args.target_model == 'roberta':
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings, batch_size=args.batch_size)
    elif args.target_model == 'bart':
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.model.shared, batch_size=args.batch_size)

    return victim


"""mask-filling"""
def mask_and_fill(data_list, label_list, victim, generator, tokenizer):
    """1. 获得mask位置，并mask"""
    outs = {}

    if args.mask_mode == 'random':
        """1.1 随机mask"""
        for idx, (text, label) in enumerate(zip(data_list, label_list)):
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')

            # 观察重要性
            # orig_probs = victim.get_prob([' '.join(text)])
            # orig_label = orig_probs.argmax(axis=1)
            # orig_prob = orig_probs.max()
            # len_text = len(text)
            # leave_1_texts = [text[:ii] + ['<oov>'] + text[min(ii + 1, len_text):] for ii in range(len_text)]
            # leave_1_probs = victim.get_prob([' '.join(lt) for lt in leave_1_texts])
            # leave_1_probs_argmax = leave_1_probs.argmax(axis=1)
            # a = (leave_1_probs_argmax != orig_label)
            # b = leave_1_probs.max(axis=1)
            # c = torch.index_select(torch.tensor(orig_probs[0]), 0, torch.tensor(leave_1_probs_argmax)).data.numpy()
            # d = b - c
            # import_scores = (orig_prob - np.squeeze(leave_1_probs[:, orig_label]) + a * d)
            # print(np.argsort(-import_scores).tolist())

            masked_flags = []
            masked_ori_words = []
            masked_texts = []
            for jj in range(min(args.max_seq_length, len(text))):
                word = text[jj]
                r_seed = np.random.rand(args.sample_size)
                n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                masked_texts.append(n)
                m = [True if rs < args.mask_ratio else False for rs in r_seed]
                masked_flags.append(m)
            masked_texts = np.array(masked_texts).T.tolist()
            masked_flags = np.array(masked_flags).T.tolist()
            for i in range(args.sample_size):
                masked_ori_words.append(np.array(text[:min(args.max_seq_length, len(text))])[masked_flags[i]].tolist())
            # 注：可能会把符号mask掉
            # 在文本开头加入prompt
            masked_texts = [prompt_templates[label] + mt for mt in masked_texts]
            # 在每句话后面加入prompt
            # for i in range(len(masked_texts)):
            #     text_lst = re.split(r"([.。!！?？；;，,])", ' '.join(masked_texts[i]))
            #     text_lst.append("")
            #     text_lst = ["".join(i) for i in zip(text_lst[0::2], text_lst[1::2])]
            #     text_lst = [tl + ' ' + ' '.join(prompt_templates[label]) for tl in text_lst]
            #     text_lst = ' '.join(prompt_templates[label]) + ' ' + ' '.join(text_lst)
            #     masked_texts[i] = text_lst.split(' ')

            outs[idx] = {'label': label, 'text': text, 'masked_texts': masked_texts, 'masked_ori_words': masked_ori_words}
    elif args.mask_mode == 'clsImp':
        """1.2 分类重要性"""
        for idx, (text, label) in enumerate(zip(data_list, label_list)):
            orig_probs = victim.get_prob([' '.join(text)])
            orig_label = orig_probs.argmax(axis=1)
            orig_prob = orig_probs.max()
            len_text = len(text)
            leave_1_texts = [text[:ii] + ['<oov>'] + text[min(ii + 1, len_text):] for ii in range(len_text)]
            leave_1_probs = victim.get_prob([' '.join(lt) for lt in leave_1_texts])
            leave_1_probs_argmax = leave_1_probs.argmax(axis=1)
            a = (leave_1_probs_argmax != orig_label)
            b = leave_1_probs.max(axis=1)
            c = torch.index_select(torch.tensor(orig_probs[0]), 0, torch.tensor(leave_1_probs_argmax)).data.numpy()
            d = b - c
            # if len(leave_1_probs.shape) == 1: # 说明该文本只有一个单词，增加一维
            #     leave_1_probs = leave_1_probs.unsqueeze(0)
            # exit(0)
            import_scores = (orig_prob - np.squeeze(leave_1_probs[:, orig_label]) + a * d)
            masked_text = text.copy()
            # print(np.argsort(-import_scores).tolist())
            # 取前mask_ratio个重要的位置进行mask（候选集只有一条）
            masked_pos = np.argsort(-import_scores).tolist()[:int(len_text*args.mask_ratio)]
            for mp in masked_pos:
                masked_text[mp] = args.mask_token

            # mask掉不重要的
            # masked_pos = np.argsort(import_scores).tolist()[:int(len_text * args.mask_ratio)]
            # for mp in masked_pos:
            #     masked_text[mp] = args.mask_token

            # 在文本开头加入prompt
            masked_text = prompt_templates[label] + masked_text

            outs[idx] = {'label': label, 'text': text, 'masked_texts': [' '.join(masked_text)]}
    elif args.mask_mode == 'selectedPS':
        pos_tagger = nltk.tag.perceptron.PerceptronTagger()
        """1.3 语义保持：避开重要位置"""
        for idx, (text, label) in enumerate(zip(data_list, label_list)):
            # 确定可mask位置
            tokens = pos_tagger.tag(text)
            candi_ps = []
            for ii, (word, ps) in enumerate(tokens):
                # print(word, ps)
                if ps.startswith('JJ'):  # 避开形容词JJ、副词RB、实义动词VB or ps.startswith('RB') or ps.startswith('VB')
                    continue
                else:
                    candi_ps.append(ii)
            if text[-1] not in stop_mark:  # 可能存在原始文本的最后一个词语不是分隔符
                text.append('.')
                candi_ps.append(len(text)-1)  # 候选位置加上最后一个分隔符
            # 进行mask
            masked_texts = []
            masked_flags = []
            masked_ori_words = []
            for jj in range(len(text)):  #######截断不合适，后面补全的时候可能可以用到后面的信息
                word = text[jj]
                if jj in candi_ps:
                    r_seed = np.random.rand(args.sample_size)
                    n = [args.mask_token if rs < args.mask_ratio else word for rs in r_seed]
                    m = [True if rs < args.mask_ratio else False for rs in r_seed]
                else:
                    n = [word] * args.sample_size
                    m = [False] * args.sample_size
                masked_texts.append(n)
                masked_flags.append(m)

            masked_texts = np.array(masked_texts).T.tolist()
            masked_flags = np.array(masked_flags).T.tolist()
            for i in range(args.sample_size):
                # masked_ori_words.append(np.array(text[:min(args.max_seq_length, len(text))])[masked_flags[i]].tolist())  # 截断
                masked_ori_words.append(np.array(text)[masked_flags[i]].tolist())  # 不截断
            # 注：可能会把符号mask掉
            # 在文本开头加入prompt
            masked_texts = [prompt_templates[label] + mt for mt in masked_texts]
            # 在每句话后面加入prompt
            # for i in range(len(masked_texts)):
            #     text_lst = re.split(r"([.。!！?？；;，,])", ' '.join(masked_texts[i])) # 符号可能被mask了，怎么办
            #     text_lst.append("")
            #     text_lst = ["".join(i) for i in zip(text_lst[0::2], text_lst[1::2])]
            #     text_lst = [tl + ' ' + ' '.join(prompt_templates[label]) for tl in text_lst]
            #     text_lst = ' '.join(prompt_templates[label]) + ' ' + ' '.join(text_lst)
            #     masked_texts[i] = text_lst.split(' ')
            outs[idx] = {'label': label, 'text': text, 'masked_texts': masked_texts, 'masked_ori_words':masked_ori_words}

            # outs[idx] = {'label':label, 'text':tokenizer.tokenize(' '.join(text)),'masked_texts':masked_texts,'masked_ori_words': masked_ori_words}

    # print(outs)

    """2. 对于所有mask位置，进行补全"""
    for idx, (text, label) in enumerate(zip(data_list, label_list)):
        masked_texts = outs[idx]['masked_texts']
        temp = []

        mydataset, _ = MyDataset(args, tokenizer).transform_text(masked_texts, [label]*len(masked_texts))
        sampler = torch.utils.data.SequentialSampler(mydataset)
        dataloader = torch.utils.data.DataLoader(mydataset, sampler=sampler, batch_size=args.batch_size)

        # 对每个batch
        for input_ids, input_mask, segment_ids, _ in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            with torch.no_grad():
                if args.target_model == 'bert':
                    prediction_scores = generator(input_ids, token_type_ids=segment_ids, attention_mask=input_mask).logits  # [b_size,128,30522]
                elif args.target_model == 'roberta' or args.target_model == 'bart':
                    prediction_scores = generator(input_ids, input_mask).logits  # [batch_size,seq_len,vocab_size]
                mask_ps = []  # list，每个list是该句子mask位置的所有索引 注：是对于处理完的input_ids来说的位置（经过token之后）
                for i in range(len(input_ids)):
                    if args.target_model == 'bert':
                        m = np.where(input_ids[i].cpu().numpy() == tokenizer.convert_tokens_to_ids([args.mask_token]))[0].tolist()
                    elif args.target_model == 'roberta' or args.target_model == 'bart':
                        m = np.where(input_ids[i].cpu().numpy() == tokenizer(args.mask_token)['input_ids'][1])[0].tolist()
                    mask_ps.append(m)
                # 每个位置保留topk
                _, pred_ids = prediction_scores.topk(k=args.topk, dim=2, largest=True, sorted=True)  # [batch_size,seq_len,k)
                # 每个位置选择第Top k的
                # pred_ids = pred_ids[:,:,2].unsqueeze(2) # top 3的
                # pred_ids = pred_ids[:,:,0].unsqueeze(2) # top 1的  #######删

                for i in range(len(input_ids)):  # 对于每条mask数据
                    mp_a = mask_ps[i]  # 对于该数据，mask的位置
                    # 存在无mask位置的情况
                    if len(mp_a) == 0:
                        continue
                    input_ids_a = input_ids[i]  # 当前数据
                    num_padding = len(input_mask[i]) - torch.sum(input_mask[i])  # 当前数据中padding位置的个数
                    candi_ids = pred_ids[i, mp_a[:], :] #.cpu().numpy().tolist()  # 对于该数据，mask位置的候选集

                    if args.word_emb:
                        """每个位置从topk里选择word embedding最相近的"""
                        candi_ids = candi_ids.cpu().numpy().tolist()
                        ori_words = outs[idx]['masked_ori_words'][i]
                        ori_word_embeddings = []
                        oov_ps = []
                        for o in range(min(len(mp_a), len(ori_words))):
                            ow = ori_words[o]
                            if ow not in args.glove_model.keys():  # 可能原始词语不在glove词汇表中
                                ori_word_embeddings.append([0.1]*200)
                                oov_ps.append(o)
                            else:
                                ori_word_embeddings.append(args.glove_model[ow])
                        ori_word_embeddings = torch.tensor(np.array(ori_word_embeddings)).repeat(1,args.topk).reshape(-1,args.topk,200) # 200是word embedding维度
                        candi_word_embeddings = []
                        for ci in candi_ids: # 对于每个位置的候选词
                            candi_words_a = tokenizer.convert_ids_to_tokens(ci)
                            # candi_word_embeddings_a = list(itemgetter(*candi_words_a)(args.glove_model))
                            candi_word_embeddings_a = []
                            for c in range(len(candi_words_a)):
                                cw = candi_words_a[c]
                                if cw not in args.glove_model.keys():  # 可能候选词语不在glove词汇表中
                                    candi_word_embeddings_a.append([0.1] * 200)  # 随便给了个emb，可能有问题
                                else:
                                    candi_word_embeddings_a.append(args.glove_model[cw])
                            candi_word_embeddings.append(candi_word_embeddings_a)
                        candi_word_embeddings = torch.tensor(np.array(candi_word_embeddings))
                        sim_cos = torch.cosine_similarity(ori_word_embeddings, candi_word_embeddings, dim=2)
                        # sim_cos = torch.sqrt(torch.sum((ori_word_embeddings - candi_word_embeddings) ** 2, dim=2))

                        sim_cos_max_id = torch.argmax(sim_cos, dim=1)
                        sim_cos_max_id[oov_ps] = 0  # 对于原词是oov的，直接取top1
                        candi_ids = np.array(candi_ids)[range(len(candi_ids)), sim_cos_max_id].reshape(len(candi_ids), 1)
                    else:
                        if args.topk == 1:
                            candi_ids = candi_ids[:, 0].cpu().numpy().reshape(len(candi_ids), 1)  # 保存每个位置的第一个候选
                        else:
                            candi_ids = candi_ids.cpu().numpy().tolist()  # 所有位置的所有候选都保存

                    candi_comb = list(itertools.product(*candi_ids))  # 对于该数据，所有mask位置候选集的可能组合
                    input_ids_aa = input_ids_a.unsqueeze(0).repeat(len(candi_comb), 1)  # 当前数据的所有生成候选数据
                    # 为所有候选数据，完成替换
                    for j in range(len(input_ids_aa)): # 对于每条候选数据
                        input_ids_aa[j][mp_a] = torch.tensor(list(candi_comb[j])).cuda()
                        # 注意：只保存和输入相对应的输出
                        # temp = input_ids_aa[j][1+6: (args.max_seq_length - num_padding) - 1].cpu().numpy().tolist()
                        # 删掉首尾的[CLS]，padding，[SEP]，以及prompt
                        tt = input_ids_aa[j][1: (args.max_seq_length - num_padding) - 1].cpu().numpy().tolist()
                        para_sents = tokenizer.convert_ids_to_tokens(tt)
                        # 将分词复原
                        para_sents = tokenizer.convert_tokens_to_string(para_sents)
                        pretok_sent = para_sents.replace(' '.join(prompt_templates[label]), '').replace('</s>','').strip().replace('  ', ' ')  # 可能会补全</s>造成，输入网络预测时有多个分隔符
                        # if args.target_model == 'roberta':
                        #     para_sents = tokenizer.convert_tokens_to_string(para_sents)
                        #     pretok_sent = para_sents.replace(' '.join(prompt_templates[label]), '').strip().replace('  ', ' ')
                        # else:
                        #     para_sents = tokenizer.convert_tokens_to_string(para_sents)
                        #     pretok_sent = para_sents.replace(' '.join(prompt_templates[label]),'').strip().replace('  ', ' ')
                        temp.append({'masked_pos': mp_a, 'para_texts': pretok_sent})
        outs[idx]['para'] = temp

    # 选择写不写入文件
    # lpt = True if args.load_prompt_trained else False
    # we = True if args.word_emb else False
    # with open('/pub/data/huangpei/TextFooler/prompt_results/outs_%s_%s_%s_%s_%.2f_%d_%d_we%s_lpt%s_phead.json' %
    #           (args.task, args.target_model, args.attack_level, args.mask_mode, args.mask_ratio, args.sample_size,
    #            args.topk, we, lpt), 'w') as fw:
    #     json.dump(outs, fw, indent=4)

    return outs


def main():
    np.random.seed(10)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.rank = 0

    """Dataset"""
    if args.task == 'trec':
        test_dataset = datasets.load_dataset("trec", split="test").map(lambda x: dataset_mapping_trec(x))
        attack_dataset = test_dataset
    else:
        # Read from files
        _, _, test_data_list, test_label_list = read_text(args.data_path, args.task)
        test_data_list = test_data_list[:100]
        test_label_list = test_label_list[:100]
        test_data = dataset_mapping(test_data_list, test_label_list)
        attack_dataset = datasets.Dataset.from_list(test_data)  # 数据中默认不包含"target"字段，导致默认为untargeted攻击（扰动后的预测不等于原本的预测）
        # train_data = dataset_mapping(train_data_list, train_label_list)
        # attack_dataset = datasets.Dataset.from_list(train_data)  # 数据中默认不包含"target"字段，导致默认为untargeted攻击（扰动后的预测不等于原本的预测）

    """Victim"""
    # model = AutoModelForSequenceClassification.from_pretrained('aychang/base-cased-trec-coarse')
    # tokenizer = AutoTokenizer.from_pretrained('aychang/bert-base-cased-trec-coarse')
    # victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, word_embeddings)
    victim = build_victim()

    # preds_test = victim.get_pred([d['x'] for d in attack_dataset])
    # print('Original test acc = %.4f' % np.mean(preds_test == np.array([d['y'] for d in attack_dataset])))
    # exit(0)

    """加载补写网络"""
    if args.target_model == 'bert':
        generator = BertForMaskedLM.from_pretrained('bert-base-uncased').cuda()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif args.target_model == 'roberta':
        generator = RobertaForMaskedLM.from_pretrained(args.model_path + 'roberta-base').cuda()
        tokenizer = RobertaTokenizer.from_pretrained(args.model_path + 'roberta-base')
    elif args.target_model == 'bart':
        generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large').cuda() # modelling_bart.py里写了Mask filling only works for bart-large
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    """生成对抗样本候选集"""
    print("Generate candidate adversarial examples using prompt...")
    outs = mask_and_fill(test_data_list, test_label_list, victim, generator, tokenizer)
    # print(outs)

    """Analyze attacking results"""
    print("Attack...")
    pred_true = 0
    correct_success = 0
    correct_failed = 0
    adv_outs = {}
    for idx in outs.keys():  # 对于每条测试样例
        # 1.对原始数据进行预测
        orig_probs = victim.get_prob([' '.join(outs[idx]['text'])])
        pred_label = orig_probs.argmax(axis=1)
        if pred_label != outs[idx]['label']:
            continue
        else:
            # 2.对预测正确的，判断是否攻击成功
            print('------', idx, outs[idx]['label'], orig_probs.max()) # , len(outs[idx]['text'])) #, outs[idx]['text'])
            pred_true += 1
            # 获得所有的改写
            para_texts = [o['para_texts'] for o in outs[idx]['para']]
            if len(para_texts) > 0:
                para_probs = victim.get_prob(para_texts)
                para_labels = para_probs.argmax(axis=1)
                if np.sum(para_labels != outs[idx]['label']) > 0:
                    correct_success += 1
                    # 保存对抗样本
                    print('Succ attack!')
                    # print('ori text: ', outs[idx]['text'])
                    # print('para text: ', para_texts)
                    if len(para_labels) == 1:  # 只有一条改写
                        adv_texts = para_texts
                    else:
                        adv_texts = np.array(para_texts)[para_labels != outs[idx]['label']].tolist()
                    adv_outs[idx] = {'label': outs[idx]['label'], 'text': ' '.join(outs[idx]['text']), 'adv_texts': adv_texts}
                    # print('adv index: ', para_labels != outs[idx]['label'])
                    # print('adv text: ', adv_texts)
                else:
                    correct_failed += 1
                    print('Fail attack!')
                    # print('ori text: ', len(outs[idx]['text']), ' '.join(outs[idx]['text']))
                    # print('ori text: ', len(outs[idx]['text']))
                    # print('para text: ', para_texts)
            else:
                correct_failed += 1
                print('Fail attack!')
                # print('ori text: ', len(outs[idx]['text']))

    # print(no_para)
    print(args.task, args.target_model_path)
    print(correct_success, correct_failed, len(test_label_list))
    print("Suc = %.4f, Rob = %.4f (#%d)" % ((correct_success / (correct_success + correct_failed)), (correct_failed / len(test_label_list)), len(test_label_list)))


if __name__ == '__main__':

    main()




