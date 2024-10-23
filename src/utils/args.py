import argparse
import os
import sys
import logging

def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory:', log_dir)
    
    return logger

def get_config():
    parser = get_public_config()
    args = parser.parse_args()
    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    args.logger = logger
    args.log_dir = log_dir
    return args

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='PEMS04')
    parser.add_argument('--years', type=str, default='2018')
    parser.add_argument('--model_name', type=str, default='stgcn')
    parser.add_argument('--seed', type=int, default=998244353)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_length', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')


    parser.add_argument('--without_backbone', action='store_true')
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--stlora', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--pre_train', type=str, default="")
    parser.add_argument('--save', type=str, default="")
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--num_mlrfs', type=int, default=4)
    parser.add_argument('--num_nalls', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--step_size',type=int, default=30)
    parser.add_argument('--gamma',type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=int, default=5)
    parser.add_argument('--adj_type', type=str, default='doubletransition')


    return parser