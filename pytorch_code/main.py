#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation, ProductData
from model import *
from datetime import datetime

seed=42

import torch
torch.manual_seed(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)



def record_training_stats(opt,model):
    import wandb
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="SR-GNN_AmazonM2_GridSearch",
        # Track hyperparameters and run metadata
        config={
        "dataset": opt.dataset,
        "batchSize": opt.batchSize,
        "hiddenSize": opt.hiddenSize,
        "epoch":opt.epoch,
        "lr":opt.lr,
        "lr_dc":opt.lr_dc,
        "l2": opt.l2,
        "gnn_propogation_steps": opt.step,
        "epoch to wait before early stop":opt.patience,
        "nonhybrid":opt.nonhybrid,
        "validation":opt.validation,
        "valid_portion":opt.valid_portion,
        "K of Recall and MRR":opt.recall_mrr_k
        },reinit=True)
    # wandb.watch((model), log='all')
    return wandb, run

def main(opt):
    product_data = pickle.load(open('../datasets/' + opt.dataset + '/filtered_products_with_features.txt', 'rb'))
    product_data = ProductData(product_data)
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/sample_train.txt', 'rb'))
    print("size of training data", (len(train_data[0]),len(train_data[1])))
    # train_data = (train_data[0][:100],train_data[1][:100])
    # opt.epoch = 1
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/sample_test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    # n_code is number of nodes + 1, in the paper, it mentioned to have 43,097 items in Diginetica and 37,483 items in yoochoose
    # that's why ".weights[1:]" in line https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/model.py#L87 and "targets - 1" in line https://github.com/CRIPAC-DIG/SR-GNN/blob/master/pytorch_code/model.py#L133
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'amazonM2':
        n_node = 607048+1
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node, product_data))
    
    if opt.evaluation_only:
        print('-----------------------Evaluation----------------------')
        model.load_state_dict(torch.load(f'../saved_model/{opt.model_name}.pt'))
        model.eval()
        _,hit, mrr = evaluation(model, test_data, recall_mrr_k=opt.recall_mrr_k)
        print('Evaluation\tRecall@%d:\t%.4f\tMMR@%d:\t%.4f'% (opt.recall_mrr_k, hit, opt.recall_mrr_k, mrr))
        print('-------------------------------------------------------')
        return
    
    if opt.record_wandb:
        run,wandb=record_training_stats(opt,model)
    else:
        wandb=None
        run=None
    start = time.time()
    start_datetime = datetime.now().strftime('%y%d%m%H%M%S')
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, wandb, opt.recall_mrr_k)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        if flag==1:
            torch.save(model.state_dict(), f'../saved_model/best_model_{start_datetime}.pt')
        print('Best Result:')
        print('\tRecall@%d:\t%.4f\tMMR@%d:\t%.4f\tEpoch:\t%d,\t%d'% (opt.recall_mrr_k, best_result[0], opt.recall_mrr_k, best_result[1], best_epoch[0], best_epoch[1]))
        if wandb is not None:
            wandb.log({"epoch": epoch, f"Validation Recall@{opt.recall_mrr_k} per epoch": hit})
            wandb.log({"epoch": epoch, f"Validation MMR@{opt.recall_mrr_k} per epoch": mrr})
            wandb.log({"epoch": epoch, f"Best Validation Recall@{opt.recall_mrr_k} per epoch": best_result[0]})
            wandb.log({"epoch": epoch, f"Best Validation MMR@{opt.recall_mrr_k} per epoch": best_result[1]})
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    if wandb is not None:
        run.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazonM2', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample/amazonM2')
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=1024, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.9, help='learning rate decay rate') # [0.5, 0.9]
    # parser.add_argument('--lr_dc_step', type=int, default=100, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.2, help='split the portion of training set as validation set')
    parser.add_argument('--evaluation_only', action='store_true', help='evaluate with testing set')
    parser.add_argument('--model_name', default='', help='model to be evaluated')
    parser.add_argument('--record_wandb', action='store_true', help='whether record training stats in wandb')
    parser.add_argument('--recall_mrr_k', type=int, default=100, help='the K of MRR@K and Recall@K')
    opt = parser.parse_args()
    print(opt)
    # Grid Search
    import itertools
    import glob
    import os
    from pathlib import Path
    lrs = [1e-3, 5e-4, 1e-4, 1e-5]
    lr_dcs = [0.5, 0.9]
    l2s = [1e-3, 1e-4, 1e-5]
    all_params_to_search = [lrs,lr_dcs,l2s]
    list_params_to_search = list(itertools.product(*all_params_to_search))
    print(len(list_params_to_search))
    for hyparam_idx,(lr,lr_dc,l2) in enumerate(list_params_to_search):
        print(f'{hyparam_idx}: ',(lr,lr_dc,l2))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        opt.lr=lr
        opt.lr_dc=lr_dc
        opt.l2=l2
        opt.validation=True
        opt.evaluation_only = False
        opt.record_wandb=True
        main(opt)
        list_of_model_files = glob.glob('../saved_model/*.pt')
        latest_model_file = max(list_of_model_files, key=os.path.getctime)
        opt.evaluation_only = True
        opt.model_name = Path(latest_model_file).stem
        opt.record_wandb=False
        main(opt)
        opt.validation=True
        opt.evaluation_only = False
        opt.record_wandb=True
    # main(opt)
