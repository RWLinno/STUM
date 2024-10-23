import os

is_debug = True
if is_debug:
    os.environ['WANDB_MODE'] = 'disabled'

import time
import argparse
import numpy as np
import torch.optim as optim
from src.__init__ import *
import src.loralib as lora
import src.lora4st as lora4st
from torchinfo import summary

def main():
    args = get_config() # Get arguments
    init_seed(args.seed) # Set random seed
    args.data_path, args.adj_path, args.node_num = get_dataset_info(args.dataset) # Get dataset info
    args.adj_mx = load_adj_from_numpy(args.adj_path) # Load adjacency matrix
    # args.adj_mx = normalize_adj_mx(args.adj_mx, args.adj_type) # Normalize adjacency matrix
    args.supports = [torch.tensor(i).to(args.device) for i in args.adj_mx] # Convert adjacency matrix to tensor

    args.loss_fn = masked_mae
    
    args.model = lora4st.STLoRA(backbone=None, args=args)
    
    if args.pre_train:
        args.load_pretrain_path = os.path.join('./experiments', args.model_name, args.dataset, args.pre_train)
        args.logger.info("load pretrain model from: ", args.load_pretrain_path)
        args.model.load_state_dict(torch.load(args.load_pretrain_path))
    
    # 仅包含需要训练的参数（即不冻结的参数）
    if args.frozen:
        lora.mark_only_lora_as_trainable(args.model)
        params_to_optimize = filter(lambda p: p.requires_grad, args.model.parameters())
    else:
        params_to_optimize = args.model.parameters()
    
    args.optimizer = torch.optim.Adam(params_to_optimize, lr=args.lrate, weight_decay=args.wdecay)
    args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, step_size=20, gamma=0.1)
    args.dataloader, args.scaler = load_dataset(args.data_path, args)
    engine = BaseEngine(args)

    #summary(engine.model, input_size=(args.batch_size, args.seq_length, args.node_num, args.input_dim)) # Print model summary
    engine.model.to(args.device)
    args.logger.info(engine.model)
    
    train_time = time.time()
    if args.mode == 'train':
        engine.train()
    elif args.mode == 'eval' or args.mode == 'test':
        engine.evaluate(args.mode)
    else:
        raise ValueError
    
    end_time = time.time()
    #print(f"total {args.mode} time: {end_time - train_time} s")
    args.trainable_params, args.all_param = print_trainable_parameters(engine.model) # Print trainable parameters
    args.logger.info(f"trainable parameters: {args.trainable_params}, all parameters: {args.all_param}")
    args.logger.info(f"total {args.mode} time: {end_time - train_time} s")
    

    if args.save:
        if not os.path.exists('./save'):
            os.makedirs('./save')
        torch.save(engine.model.state_dict(), './save/'+args.save)

    #print(f"model:{args.model_name}, dataset:{args.dataset}, mode:{args.mode}, ST-LoRA Enhancement:{args.stlora}, seed:{args.seed}")
    #print(" finished!! thank you!!")

    args.logger.info(f"model:{args.model_name}, dataset:{args.dataset}, mode:{args.mode}, ST-LoRA Enhancement:{args.stlora}, seed:{args.seed}")
    args.logger.info(f'mlp:{args.mlp}, fronzen:{args.frozen}, pre_train:{args.pre_train}, save:{args.save}')
    args.logger.info(" finished!! thank you!!")

if __name__ == "__main__":
    main()