import argparse
import os
# 覆盖tensorflow安装warning输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument("--target_model_path", type=str, default=None)
parser.add_argument("--data_path", type=str, default="/pub/data/huangpei/PAT-AAAI23/TextFooler/data/", help="path of data")
parser.add_argument("--save_path", type=str, default="/pub/data/huangpei/PAT-AAAI23/TextFooler/models/", help="path of saved code/project")
parser.add_argument("--task", type=str, default='mr', help="task name: mr/imdb/snli")
parser.add_argument("--target_model", type=str, default='bert', help="bert or roberta")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size to get prediction")
parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
parser.add_argument('--sep_token', type=str, default='[SEP]')  # For snli: [SEP] for BERT, </s></s> for ROBERTA
parser.add_argument('--mask_token', type=str, default='[MASK]')  #  [MASK] for BERT, <mask> ROBERTA
parser.add_argument('--syn_num', type=int, default=5)

# ensemble
parser.add_argument('--num_models', type=int, default=3)
parser.add_argument('--lamda', type=float, default=0, help='weight of ensemble entropy loss')
parser.add_argument('--log_det_lamda', type=float, default=0, help='weight of pred det loss')
parser.add_argument('--log_det_att', type=float, default=0, help='weight of att det loss')
parser.add_argument('--lamda_att', type=float, default=0, help='weight of att det loss')
parser.add_argument('--num_param', type=int, default=2363905)  # 109483778 为bert参数个数，2363905 为第0层self-attention的参数
parser.add_argument('--aux_weight', type=float, default=0, help='if>0, perturb hidden states with this wight of aux model')
parser.add_argument('--perturb_attention', action='store_true', help='if use, perturb attention matrix')
parser.add_argument('--modif_att_layer', type=int, default=0)


# ASCC/DNE
parser.add_argument('--ascc', action='store_true', help='if use, get syn data in data_utils')
parser.add_argument('--dne', action='store_true', help='if use, get syn of syn data in data_utils')

parser.add_argument('--attack_alg', type=str, default=None)


# args = parser.parse_args()
args =parser.parse_known_args()[0]

args.data_path += args.task
nclasses_dict = {'imdb': 2, 'mr': 2, 'snli': 3, 'trec': 6}
args.num_labels = nclasses_dict[args.task]
seq_len_list = {'imdb': 256, 'mr': 128, 'snli': 128, 'trec': 128}
args.max_seq_length = seq_len_list[args.task]
if args.target_model == 'roberta' :
    args.sep_token = '</s></s>'
    args.mask_token = '<mask>'
if args.target_model == 'bart':
    args.sep_token = '</s>'
    args.mask_token = '<mask>'

# args.max_seq_length = 3


