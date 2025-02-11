#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm

import numpy as np 
import pandas as pd 

import torch
from torch.utils.data import DataLoader

from utilities.adversarial_loss import GANLoss
from utilities.dataset import DatasetReMIND
from utilities.utils import (
    create_logger, 
    infinite_iterable,
    save,
    set_determinism)

from network.mhvae import MHVAE2D
from network.mvae import MVAE2D
from itertools import chain, combinations

# from monai.utils import set_determinism
import nibabel as nib


PHASES = ['validation', 'inference']



def run_inference(paths_dict, saving_path, model, device, logger, opt):
    logger.info(f"Modalities are {opt.modalities}")

    # Define transforms for data normalization and augmentation
    dataloaders = dict()
    subjects_dataset = dict()
    ind_batch = dict()

    for phase in PHASES:
        paths_dict['unnormalized'][phase] = {mod:paths_dict['unnormalized'][phase][mod] for mod in opt.modalities}
        paths_dict['normalized'][phase] = {mod:paths_dict['normalized'][phase][mod] for mod in opt.modalities}
        ind_batch[phase] = np.arange(0, len(paths_dict['normalized'][phase][opt.modalities[0]]),  opt.batch_size)
        subjects_dataset[phase] = DatasetReMIND(
            paths_unnorm=paths_dict['unnormalized'][phase], 
            paths_norm=paths_dict['normalized'][phase],
            mode=phase, normalization=True, type_normalization=opt.type_normalization)
        dataloaders[phase] = infinite_iterable(DataLoader(subjects_dataset[phase], batch_size=opt.batch_size, shuffle=phase=='training', num_workers=opt.workers))

    # Training parameters 
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    list_f = [eval(k.replace('.pth','').split('_')[2]) for k in os.listdir(os.path.join(opt.model_dir, 'models')) if 'main' in k and not 'final' in k and not 'best' in k]
    epoch_inf = max(list_f)
    if not opt.epoch_inf==1000:
        epoch_inf = opt.epoch_inf
    model.load_state_dict(torch.load(save_path.format('main',epoch_inf)))
    logger.info(f"Loading model from {save_path.format('main',epoch_inf)}")
    first_mod = opt.modalities[0]
    
    assert opt.batch_size ==1, 'Only batch_size==1 supported'
    model.eval()  # Set model to evaluate mode

    for phase in PHASES:
        # Iterate over data
        for _ in tqdm(ind_batch[phase]):
            batch = next(dataloaders[phase])
            name = batch[f"{first_mod}_name"][0]
            imgs = dict()
            imgs_norm = dict()
            nonempty_list = []
            for mod in opt.modalities:
                imgs[mod] = batch[mod].to(device).permute(0,4,1,2,3)
                imgs[mod] = imgs[mod].reshape(-1, 1, *batch[mod].shape[2:4])
                imgs_norm[mod] = batch[f"{mod}_norm"].to(device).permute(0,4,1,2,3)
                imgs_norm[mod] = imgs_norm[mod].reshape(-1, 1, *batch[f"{mod}_norm"].shape[2:4])
                if not torch.all(imgs[mod]==-1):
                    nonempty_list.append(mod)

            subset_us = ['us']
            subset_mr = [k for k in nonempty_list if not 'us'==k]
            total_subsets_mr = list(chain.from_iterable(combinations(subset_mr, r) for r in range(1,len(subset_mr)+1)))
            with torch.no_grad():      
                affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
                name = batch[f"{first_mod}_name"]
                
                name = batch[f"{first_mod}_name"][0].replace(first_mod,'')+'{}_{}.nii.gz'
                
                features = []
                paths_features = []    
                
                # for temp in [0.1, 0.5, 1]:
                for temp in [0.7]:
                # for temp in [0.5]:
                    # Save us
                    pred, _, feat  = model({'us':imgs['us'].clone()}, temp, return_feat=True, return_cat=True)
                    save(pred, affine, os.path.join(saving_path, name.format('us',temp)))
                    
                    # Save MR
                    for subset in total_subsets_mr:
                        pred, _, feat  = model({mod:imgs[mod].clone() for mod in subset}, temp, return_feat=True, return_cat=True)
                        listToStr = '_'.join([str(elem) for elem in subset])
                        save(pred, affine, os.path.join(saving_path,name.format(listToStr, temp)))
                gt = torch.cat([imgs_norm[mod] for mod in opt.modalities],1)    
                save(gt, affine, os.path.join(saving_path, name.format('gt','img')))
                        


def main():
    opt = parsing_data()

    # FOLDERS
    fold_dir = opt.model_dir
    
    output_path = os.path.join(fold_dir,f'inference_multitemp')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_path_logs = os.path.join(fold_dir,'logs')
    if not os.path.exists(output_path_logs):
        os.makedirs(output_path_logs)

    logger = create_logger(output_path_logs)
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Number of modalities: {len(opt.modalities)}")
    logger.info(f"With residual: {opt.no_res}")
    logger.info(f"With se: {opt.no_se}")
    logger.info(f'GT path: {opt.path_data_norm}')
    set_determinism(seed=opt.seed)

    if torch.cuda.is_available():
        logger.info('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        
    logger.info("[INFO] Reading data")
    # PHASES
    split_path = os.path.join(opt.dataset_split)

    df_split = pd.read_csv(split_path,header =None)
    list_file = dict()
    for phase in PHASES: # list of patient name associated to each phase
        list_file[phase] = df_split[df_split[1].isin([phase])][0].tolist()

    paths_dict = dict()
    paths_dict['unnormalized'] = {phase:{modality:[] for modality in opt.modalities} for phase in PHASES}
    paths_dict['normalized'] = {phase:{modality:[] for modality in opt.modalities} for phase in PHASES}
    for phase in PHASES:
        for subject in list_file[phase]:
            if os.path.exists(opt.path_data+subject+opt.modalities[0]+'.nii.gz'):
                for modality in opt.modalities:
                    paths_dict['unnormalized'][phase][modality].append(opt.path_data+subject+modality+'.nii.gz')
                    paths_dict['normalized'][phase][modality].append(opt.path_data_norm+subject+modality+'.nii.gz')


    # MODEL 
    if opt.type_model=='mvae':
        logger.info("[INFO] Building non-hierarchical multi-modal model")  
        model = MVAE2D(
        modalities=opt.modalities,   
        base_num_features=opt.base_features,   
        num_pool=opt.pools,   
        original_shape=opt.spatial_shape[:2],
        max_features=opt.max_features,
        with_residual=opt.no_res,
        with_se=opt.no_se,
        nb_finalblocks=opt.nb_finalblocks,
        nfeat_finalblock=opt.nfeat_finalblock,
        logger=logger,
        ).to(device)
        
        
    elif opt.type_model=='mhvae':
        logger.info("[INFO] Building hierarchical multi-modal model")  
        model = MHVAE2D(
            modalities=opt.modalities,   
            base_num_features=opt.base_features,   
            num_pool=opt.pools,   
            original_shape=opt.spatial_shape[:2],
            max_features=opt.max_features,
            with_residual=opt.no_res,
            with_se=opt.no_se,
            nb_finalblocks=opt.nb_finalblocks,
            nfeat_finalblock=opt.nfeat_finalblock,
            logger=logger,
            ).to(device)
        
    else:
        raise ValueError('Wrong type_model')
        
    

    logger.info("[INFO] Running inference")
    
    run_inference(
        paths_dict, 
        output_path,
        model, 
        device, 
        logger,
        opt)



def parsing_data():
    parser = argparse.ArgumentParser(
        description='Inference using MMHVAE')

    parser.add_argument('--model_dir',
                        type=str,
                        help='Save model directory')

    parser.add_argument('--dataset_split',
                        type=str,
                        default='./splits/split.csv',
                        help='Split file (.csv)')

    parser.add_argument('--path_data',
                        type=str,
                        default='../data/unnorm/',
                        help='Path to the unnorm dataset')

    parser.add_argument('--path_data_norm',
                        type=str,
                        default='../data/unnorm/',
                        help='Path to the norm dataset')
    
    parser.add_argument('--type_normalization',
                        type=str,
                        default='min-max',
                        help='Type of normalization')
    
    parser.add_argument('--seed',
                    type=int,
                    default=3)

    parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help='Batch size: 3d volume')

    parser.add_argument('--max_features',
                    type=int,
                    default=128,
                    help='Max Latent Dimensionality')

    parser.add_argument('--base_features',
                    type=int,
                    default=16,
                    help='Latent Dimensionality highest level (divided by 2)')
    
    parser.add_argument('--nb_finalblocks',
                    type=int,
                    default=6,
                    help='Number of blocks from z_1 to x_i')
    
    parser.add_argument('--nfeat_finalblock',
                    type=int,
                    default=8,
                    help='Number of channels in blocks from z_1 to x_i')

    parser.add_argument('--pools',
                    type=int,
                    default=6,
                    help='Number of latent representation below z_1')

    parser.add_argument('--epoch_inf',
                    type=int,
                    default=1000,
                    help='Epoch used for inference')

    parser.add_argument('--workers',
                    type=int,
                    default=10,
                    help='Number of workers')

    parser.add_argument('--spatial_shape',
                    type=int,
                    nargs="+",
                    default=(192,192,160))

    parser.add_argument('--modalities',
                    type=str,
                    nargs="+",
                    default=['us', 't2'])

    parser.add_argument('--no_se',
                        action='store_false', 
                        help='Squeeze and Excitation disabled')

    parser.add_argument('--no_res', 
                        action='store_false', 
                        help='Residual connection disabled')
    
    parser.add_argument('--type_model', 
                        type=str,
                        default='mhvae',
                        )
    
    parser.add_argument('--save_images', 
                        action='store_true', 
                        help='Save images')
    
    parser.add_argument('--save_features', 
                        action='store_true', 
                        help='Save features')
    
    parser.add_argument('--save_samples',
                        action='store_true', 
                        help='Save samples')



    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()


