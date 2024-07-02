import OpenAttack_.OpenAttack as oa
import datasets
import numpy as np
import torch,os
import torch.nn.functional as F
import json

"""from shield"""
from shield.model import BertClassifierDARTS
from shield.utils import MyClassifier

"""from mine"""
from data_utils import read_text, dataset_mapping, dataset_mapping_trec, get_syn, MyDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, RobertaTokenizer, BertTokenizer, BartTokenizer
from config import args
from models import EnsembleBERT, EnsembleROBERTA


"""对于DNE，当攻击时，用凸包内点的集成作为最终输入，进行预测"""
class DNEWrapper(oa.victim.Classifier):
    def __init__(self, model : oa.victim.Classifier):
        self.model = model

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        def get_comb(p, syn):
            return (p * syn.detach()).sum(-2)

        b_size = 10
        logits = []
        for idx in range((len(input_)+b_size-1)//b_size):
            sents = input_[idx*b_size:(idx+1)*b_size]

            candidate_sents_all_1hop, candidate_labels_all_1hop, syn_flag_all_1hop = get_syn(sents, labels=[0]*len(input_))
            tokenizer_ = AutoTokenizer.from_pretrained(args.target_model_path)
            syn_ids_all, syn_input_mask_all, syn_segment_ids_all, _ = MyDataset(args, tokenizer_).convert_examples_to_features(candidate_sents_all_1hop, candidate_labels_all_1hop)
            syn_ids_all = torch.tensor(syn_ids_all, dtype=torch.long).cuda(args.rank)

            syn_segment_ids_all = syn_segment_ids_all[::args.syn_num]
            syn_input_mask_all = syn_input_mask_all[::args.syn_num]
            syn_input_mask_all = torch.tensor(syn_input_mask_all, dtype=torch.long).cuda(args.rank)
            syn_segment_ids_all = torch.tensor(syn_segment_ids_all, dtype=torch.long).cuda(args.rank)
            syn_flag_all_1hop = torch.tensor(syn_flag_all_1hop, dtype=torch.long).cuda(args.rank)
            word_embedding_syn = self.model.model.bert.embeddings.word_embeddings(syn_ids_all)  # [b_size*syn_num, seq_len, h_size]

            text_len = syn_ids_all.shape[1]
            batch_size = int(syn_ids_all.shape[0]/args.syn_num)
            syn_num_each_pos = torch.sum(syn_flag_all_1hop.view(args.syn_num, -1, args.max_seq_length), dim=0)  # [b_size, seq_len] 标志每个位置的同义词个数

            logits_adv_all = 0.0
            weight_all = []
            K = 3
            for _ in range(K):
                w = torch.zeros([batch_size, text_len, args.syn_num, 1], dtype=torch.float).cuda(args.rank)
                for i in range(batch_size):
                    for j in range(text_len):
                        syn_num_a = syn_num_each_pos[i][j].cpu().numpy().tolist()
                        alpha = [syn_num_a] * syn_num_a
                        diri = np.random.dirichlet(alpha, 1)
                        w[i, j, :syn_num_a, 0] = torch.from_numpy(diri)
                        diri = []
                embd_adv = get_comb(F.softmax(w, dim=-2), word_embedding_syn.reshape(batch_size, text_len, args.syn_num, -1))  # [b_size, seq_len, h_size]
                logit_adv = self.model.model(token_type_ids=syn_segment_ids_all, attention_mask=syn_input_mask_all, inputs_embeds=embd_adv).logits
                logits_adv = torch.nn.functional.softmax(logit_adv, dim=-1)  # [b_size, num_labels]
                logits_adv_all += logits_adv
            logits_adv_all = logits_adv_all / K
            logits.extend(logits_adv_all.cpu().detach().numpy().tolist())
        logits = np.array(logits)
        return logits


model_types = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 'bart': 'facebook/bart-base'}
model_type = model_types[args.target_model]
def build_victim():
    if args.target_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.target_model_path)
    elif args.target_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.target_model_path)
    elif args.target_model == 'bart':
        tokenizer = BartTokenizer.from_pretrained(args.target_model_path)

    if 'esb' in args.target_model_path:
        config = AutoConfig.from_pretrained(args.target_model_path)
        config.aux_weight = args.aux_weight
        config.num_models = args.num_models
        config.perturb_attention = args.perturb_attention
        checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
        new_checkpoint = {}
        for name, param in checkpoint.items():
            name_ = name.replace('bert.bert', 'bert').replace('bert.classifier', 'classifier')
            new_checkpoint[name_] = param
        if args.target_model == 'bert':
            model = EnsembleBERT.from_pretrained(model_type, state_dict=new_checkpoint, config=config)
        elif args.target_model == 'roberta':
            model = EnsembleROBERTA.from_pretrained(model_type, state_dict=new_checkpoint, config=config)
        if args.target_model == 'bert':
            word_embeddings = model.bert.embeddings.word_embeddings
        else:
            word_embeddings = model.roberta.embeddings.word_embeddings
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, word_embeddings, batch_size=args.batch_size)
    elif 'shield' in args.target_model_path:
        training_temp = 1.0
        device = 'cuda:0'
        checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
        if 'bert_layer.encoder.aux.dense.weight' in checkpoint.keys():
            del checkpoint['bert_layer.encoder.aux.dense.weight']
        if 'bert_layer.encoder.aux.dense.bias' in checkpoint.keys():
            del checkpoint['bert_layer.encoder.aux.dense.bias']
        model = BertClassifierDARTS(model_type=model_type, freeze_bert=True, output_dim=args.num_labels, ensemble=1, N=args.num_models, temperature=training_temp,
                                    gumbel=1, scaler=1, darts=True, device=device)
        model.load_state_dict(checkpoint)
        victim = MyClassifier(model, tokenizer, max_len=args.max_seq_length, device=device, batch_size=args.batch_size)
    elif ('adv' in args.target_model_path) or ('ascc' in args.target_model_path) or ('dne' in args.target_model_path):
        config = AutoConfig.from_pretrained(args.target_model_path)
        checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
        new_checkpoint = {}
        for name, param in checkpoint.items():
            name_ = name.replace('model.', '')
            new_checkpoint[name_] = param
        model = AutoModelForSequenceClassification.from_pretrained(model_type, state_dict=new_checkpoint, config=config)
        if args.target_model == 'bert':
            word_embeddings = model.bert.embeddings.word_embeddings
        else:
            word_embeddings = model.roberta.embeddings.word_embeddings
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, word_embeddings, batch_size=args.batch_size)
    if 'pat' in args.target_model_path:
        checkpoint = torch.load(args.target_model_path + '/pytorch_model.bin')
        new_checkpoint = {}
        for name, param in checkpoint.items():
            name_ = name.replace('model.bert.', '').replace('model.classifier', 'classifier')
            new_checkpoint[name_] = param
        model = AutoModelForSequenceClassification.from_pretrained(model_type, state_dict=new_checkpoint)

        if args.target_model == 'bert':
            word_embeddings = model.bert.embeddings.word_embeddings
        else:
            word_embeddings = model.roberta.embeddings.word_embeddings
        victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, word_embeddings, batch_size=args.batch_size)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.target_model_path, num_labels=args.num_labels)

        if args.target_model == 'bert':
            victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings, batch_size=args.batch_size)
        elif args.target_model == 'roberta':
            victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings, batch_size=args.batch_size)
        elif args.target_model == 'bart':
            victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, model.model.shared, batch_size=args.batch_size)

    if args.dne:
        victim = DNEWrapper(victim, batch_size=args.batch_size)

    return victim


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.rank = 0

    """Dataset"""
    if args.task == 'trec':
        test_dataset = datasets.load_dataset("trec", split="test").map(lambda x: dataset_mapping_trec(x))
        attack_dataset = test_dataset
    else:
        # Read from files
        train_data_list, train_label_list, test_data_list, test_label_list = read_text(args.data_path, args.task)
        # test_data_list = ['i like this movie , it is really overnice !']
        # test_label_list = [1]
        test_data = dataset_mapping(test_data_list, test_label_list)
        attack_dataset = datasets.Dataset.from_list(test_data)  # 数据中默认不包含"target"字段，导致默认为untargeted攻击（扰动后的预测不等于原本的预测）
        # train_data = dataset_mapping(train_data_list, train_label_list)
        # attack_dataset = datasets.Dataset.from_list(train_data)  # 数据中默认不包含"target"字段，导致默认为untargeted攻击（扰动后的预测不等于原本的预测）

    """Victim"""
    # model = AutoModelForSequenceClassification.from_pretrained('aychang/base-cased-trec-coarse')
    # tokenizer = AutoTokenizer.from_pretrained('aychang/bert-base-cased-trec-coarse')
    # victim = oa.victim.classifiers.TransformersClassifier(model, tokenizer, word_embeddings)
    victim = build_victim()

    # 测准确率
    preds_test = victim.get_pred([d['x'] for d in attack_dataset])
    print('Original test acc = %.4f' % np.mean(preds_test == np.array([d['y'] for d in attack_dataset])))
    # exit(0)

    num_list = {'imdb': 100, 'mr': 100, 'snli': 500, 'trec': 500}
    attack_num = num_list[args.task]
    # attack_num = int(0.25*len(attack_dataset))
    # attack_num = 10000
    attack_dataset = attack_dataset.select(range(attack_num))


    """Attacker"""
    # DeepWordBugAttacker, TextFoolerAttacker, PSOAttacker, BERTAttacker, SCPNAttacker
    if args.attack_alg == 'deepwordbug':
        attacker = oa.attackers.DeepWordBugAttacker()
    elif args.attack_alg == 'textfooler':
        attacker = oa.attackers.TextFoolerAttacker()
    elif args.attack_alg == 'pso':
        attacker = oa.attackers.PSOAttacker()
    elif args.attack_alg == 'bertattack':
        attacker = oa.attackers.BERTAttacker()
    elif args.attack_alg == 'scpn':
        attacker = oa.attackers.SCPNAttacker()
    elif args.attack_alg == 'viper':
        attacker = oa.attackers.VIPERAttacker()

    """Attacking..."""
    attack_eval = oa.AttackEval(attacker, victim, metrics=[oa.metric.ModificationRate(), oa.metric.SemanticSimilarity()])
    results, _ = attack_eval.eval(attack_dataset, visualize=True)

    """Analyze attacking results"""
    labels = []
    pred_orgs = []
    pred_gens = []
    correct_success, correct_failed, wrong_success, wrong_failed = 0, 0, 0, 0
    # 过滤
    sim_theo = 0.7  # 0 0.7
    if args.attack_alg == 'scpn':
        modif_theo = 1  # 对于scpn，词语级别改动较大，以modif_rate过滤意义不大
    else:
        modif_theo = 0.25  # 1 0.25
    # 不过滤
    # sim_theo = 0
    # modif_theo = 100
    # 对于roberta
    # 过滤：调节阈值
    # sim_theo = 0.3
    # modif_theo = 100

    # save_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/train_set/%s_%s_%s.json' % (args.task, args.target_model, args.attack_alg)
    # fw = open(save_path, 'w')
    for i, res in enumerate(results):
        labels.append(attack_dataset[i]['y'])
        pred_org = np.argmax(res[1])
        pred_gen = res[3]
        # gen = res[2]
        # print(i, test_dataset[i]['y'], pred_org, np.argmax(pred_gen), gen)
        pred_orgs.append(pred_org)
        pred_gens.append(np.argmax(pred_gen))

        if pred_org == attack_dataset[i]['y']:
            if res[4]['Succeed'] == True:
                modif_rate = res[4]['Word Modif. Rate']
                semantic_sim = res[4]['Semantic Similarity']
                if semantic_sim > sim_theo and modif_rate < modif_theo:
                    correct_success += 1
                    # out = {'orig_text': res[0], 'adv_text': res[2], 'label': attack_dataset[i]['y'], 'modif_rate': float(modif_rate), 'semantic_sim': float(semantic_sim)}
                    # fw.write(json.dumps(out))
                    # fw.write('\n')
                    # fw.flush()
                else:
                    correct_failed += 1
            else:
                correct_failed += 1
        elif pred_org != attack_dataset[i]['y']:
            if res[4]['Succeed'] == True:
                modif_rate = res[4]['Word Modif. Rate']
                semantic_sim = res[4]['Semantic Similarity']
                if semantic_sim > sim_theo and modif_rate < modif_theo:
                    wrong_success += 1
                else:
                    wrong_failed += 1
            else:
                wrong_failed += 1
    # fw.close()
    print(args.task, args.target_model_path, attacker)
    print(correct_success, correct_failed, wrong_success, wrong_failed, len(labels))
    print("Suc = %.4f, Rob = %.4f (#%d)" % ((correct_success / (correct_success + correct_failed)), (correct_failed / len(labels)), len(labels)))


if __name__ == '__main__':
    # save_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/test_set/%s_%s_%s.json' % (args.task, args.target_model, args.attack_alg)
    # advs = open(save_path, 'r').readlines()
    # semantic_sim = []
    # for i in range(len(advs)):
    #     adv = json.loads(advs[i])
    #     semantic_sim.append(adv['semantic_sim'])
    # semantic_sim = np.array(semantic_sim)
    # # print(np.percentile(semantic_sim, 25), np.percentile(semantic_sim, 50), np.percentile(semantic_sim, 75))
    # # exit(0)
    #
    # save_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/test_set/%s_%s_esb_%s.json' % (args.task, args.target_model, args.attack_alg)
    # advs = open(save_path, 'r').readlines()
    # semantic_sim_ = []
    # for i in range(len(advs)):
    #     adv = json.loads(advs[i])
    #     semantic_sim_.append(adv['semantic_sim'])
    # semantic_sim_ = np.array(semantic_sim_)
    #
    # for theo in np.arange(0,1,0.1):
    #     if np.mean(np.where(semantic_sim_>theo)) > np.mean(np.where(semantic_sim>theo)):
    #         print(theo, np.mean(np.where(semantic_sim_>theo)), np.mean(np.where(semantic_sim>theo)))
    # exit(0)


    main()




