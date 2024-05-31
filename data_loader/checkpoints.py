# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:02:01 2022
https://gist.github.com/rachit221195/492768a992fa2f69c0d9769f18291855#file-save_ckp-py
https://gist.github.com/rachit221195/91d5b6e96f5d268af8842235529f88c2#file-load_ckp-py
@author: walte
"""

import torch
import shutil
from glob import glob
import os

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    ckpt_name, best_model_name = get_ckp_names(state['epoch'])
    f_path = checkpoint_dir + ckpt_name
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model_name
        shutil.copyfile(f_path, best_fpath)

def save_quantized_ckp(state, is_best, checkpoint_dir, best_model_dir):
    _, ext = os.path.splitext(checkpoint_dir)
    basefname = checkpoint_dir[:-(len(ext))]
    f_path = basefname + "_quant" + ext
    torch.save(state, f_path)
    

def load_ckp_nets_only(checkpoint_fpath, netG, device):
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    netG.load_state_dict(checkpoint['state_dict'])
    return netG

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_loss = checkpoint['best_loss']
    return model, optimizer, scheduler, checkpoint['epoch'], best_loss

def load_pretrained_netG(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_model(checkpoint_fpath, model, device):
    checkpoint = torch.load(checkpoint_fpath, map_location = torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_ckp_names(epoch):
    if epoch >=0 and epoch<10:
        ckpt_name = '/checkpoint_00'+str(epoch)+'.pt'
        best_model_name = '/best_model_00'+str(epoch)+'.pt'
    elif epoch>=10 and epoch<100:
        ckpt_name = '/checkpoint_0'+str(epoch)+'.pt'
        best_model_name = '/best_model_0'+str(epoch)+'.pt'
    else:
        ckpt_name = '/checkpoint_'+str(epoch)+'.pt'
        best_model_name = '/best_model_'+str(epoch)+'.pt'
    return ckpt_name, best_model_name

def get_last_ckp_path(config):
    path = config['ckp_path']
    models_filepath =  path + '/*.pt'    
    model_list = glob(models_filepath)
    num_models = len(model_list)
    last_ckp_path=model_list[num_models-1]
    return last_ckp_path

def get_last_ckp_path_preprocessing(config):
    path = config['ckp_path_preprocessing']
    models_filepath =  path + '/*.pt'    
    model_list = glob(models_filepath)
    num_models = len(model_list)
    last_ckp_path=model_list[num_models-1]
    return last_ckp_path

def get_best_ckp_path(config):
    path = config.model_path
    return path

def get_best_ckp_path_old(config):
    path = config.model_path
    models_filepath =  path + '/*.pt'    
    model_list = glob(models_filepath)
    num_models = len(model_list)
    last_ckp_path=model_list[num_models-1]
    return last_ckp_path