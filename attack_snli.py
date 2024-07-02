import OpenAttack_.OpenAttack as oa
import datasets
from transformers import AutoTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertConfig, AutoConfig, AutoModelForSequenceClassification
import numpy as np
import torch,os
import pickle
import json
from data_utils import read_text
from config import args
from models import eval_model
from attack import build_victim
from data_utils import MyDataset

"""Refer to https://github.com/thunlp/OpenAttack/blob/master/examples/nli_attack.py"""

class NLIWrapper(oa.victim.Classifier):
    def __init__(self, model : oa.victim.Classifier):
        self.model = model

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        ref = self.context.input["hypothesis"]
        input_sents = [sent + args.sep_token + ref for sent in input_]  # roberta: </s></s>
        return self.model.get_prob(
            input_sents
        )


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.rank = 0
    # target_model_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/models/bert/snli'
    data_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/'
    task = 'snli'
    data_path += task

    """Dataset"""
    train_data_list, train_label_list, test_data_list, test_label_list = read_text(data_path, task)

    test_data = []
    for sen, l in zip(test_data_list, test_label_list):
        s1, s2 = ' '.join(sen).split(args.sep_token)
        test_data.append({'x': s1, 'hypothesis': s2, 'y': l})
    test_dataset = datasets.Dataset.from_list(test_data)  # 数据中默认不包含"target"字段，导致默认为untargeted攻击（扰动后的预测不等于原本的预测）
    attack_num = 100
    attack_dataset = test_dataset.select(range(attack_num))

    # train_data = []
    # for sen, l in zip(train_data_list, train_label_list):
    #     s1, s2 = ' '.join(sen).split(args.sep_token)
    #     train_data.append({'x': s1, 'hypothesis': s2, 'y': l})
    # train_dataset = datasets.Dataset.from_list(train_data)  # 数据中默认不包含"target"字段，导致默认为untargeted攻击（扰动后的预测不等于原本的预测）
    # attack_num = 10000 # int(0.25 * len(train_dataset))
    # attack_dataset = train_dataset.select(range(attack_num))

    """Victim"""
    victim = build_victim()
    victim = NLIWrapper(victim)

    """Attacker"""
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
    attack_eval = oa.AttackEval(attacker, victim, metrics=[oa.metric.ModificationRate(), oa.metric.SemanticSimilarity()]) # metrics=[oa.metric.EditDistance(), oa.metric.ModificationRate()]
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
    # sim_theo = 0.86
    # modif_theo = 100

    # save_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/test_set/%s_%s_%s.json' % (args.task, args.target_model, args.attack_alg)
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
                    # out = {'orig_text': res[0], 'adv_text': res[2], 'label': attack_dataset[i]['y'], 'hypothesis': attack_dataset[i]['hypothesis'],
                    #        'modif_rate': float(modif_rate), 'semantic_sim': float(semantic_sim)}
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


    # with open(save_path, 'w') as f:
    #     json.dump(outs, f)


if __name__ == "__main__":
    # save_path = '/pub/data/huangpei/PAT-AAAI23/TextFooler/data/adv_results/test_set/%s_%s_%s.json' % (args.task, args.target_model, args.attack_alg)
    # advs = open(save_path, 'r').readlines()
    # semantic_sim = []
    # for i in range(len(advs)):
    #     adv = json.loads(advs[i])
    #     semantic_sim.append(adv['semantic_sim'])
    # semantic_sim = np.array(semantic_sim)
    # print(np.percentile(semantic_sim, 25), np.percentile(semantic_sim, 50), np.percentile(semantic_sim, 75))
    # exit(0)

    main()


