#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import argparse
import time
import os
from tqdm import tqdm
from itertools import chain, combinations
import numpy as np
import pandas as pd 

import torch
from torch.utils.data import DataLoader

# Utilities
from utilities.adversarial_loss import GANLossDictV2
from utilities.dataset import DatasetReMIND
from utilities.utils import (
    create_logger, 
    infinite_iterable,
    init_training_variables,
    set_determinism,
    save_training)
from utilities.loss import l1_loss
from utilities.image_pool import ImagePool

# Network
from network.mhvae import MHVAE2D
from network.mvae import MVAE2D
from network.discriminator_pathgan2D import NLayerDiscriminator


PHASES = ['training', 'validation']

# Moving average validation
val_eval_criterion_alpha = 0.95


     
def train(paths_dict, model, netD, device, logger, opt):
    
    since = time.time()

    logger.info(f"Modalities are {opt.modalities}")
    # Define transforms for data normalization and augmentation
    dataloaders = dict()
    subjects_dataset = dict()
    ind_batch = dict()

    for phase in PHASES:
        paths_dict['unnormalized'][phase] = {mod:paths_dict['unnormalized'][phase][mod] for mod in opt.modalities}
        paths_dict['normalized'][phase] = {mod:paths_dict['normalized'][phase][mod] for mod in opt.modalities}
        ind_batch[phase] = np.arange(0, len(paths_dict['normalized'][phase][opt.modalities[0]]),  1) #only one image currently supported
        subjects_dataset[phase] = DatasetReMIND(
            paths_unnorm=paths_dict['unnormalized'][phase], 
            paths_norm=paths_dict['normalized'][phase],
            mode=phase, normalization=True, type_normalization=opt.type_normalization)
        dataloaders[phase] = infinite_iterable(DataLoader(subjects_dataset[phase], batch_size=1, shuffle=phase=='training', num_workers=opt.workers))

    # Training parameters 
    df_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    df, val_eval_criterion_MA, best_val_eval_criterion_MA, best_epoch, epoch, initial_lr = init_training_variables(model, netD, opt, logger)
    assert epoch<opt.epochs-1

    # Optimisation policy main model
    optimizer = torch.optim.Adamax(model.parameters(), lr=opt.ini_lr, betas=(0.9, 0.999,))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.epochs, eta_min=1e-4)
    
    for _ in range(epoch):
        lr_scheduler.step()

    # Optimisation policy discriminators
    lr0_disct = 2e-4
    d_optimizer = torch.optim.Adam([{'params': netD[mod].parameters()} for mod in opt.modalities], lr=lr0_disct)

    gan = GANLossDictV2('lsgan').to(device)
    first_mod = opt.modalities[0]

    continue_training = True
    
    real_pool = {mod:ImagePool(32, 16) for mod in opt.modalities}

    while continue_training:
        epoch+=1
        logger.info('-' * 10)
        logger.info('Epoch {}/'.format(epoch))
    
        if epoch==opt.warmup_value_disc-10:
            logger.info('Reinitialization optimizer {}/'.format(epoch))
            optimizer = torch.optim.Adamax(model.parameters(), lr=opt.ini_lr, betas=(0.9, 0.999,))
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, opt.epochs+1, eta_min=1e-4)
            for _ in range(epoch):
                lr_scheduler.step()

        for param_group in optimizer.param_groups:
            logger.info("Current learning rate for main model is: {}".format(param_group['lr']))

        for param_group in d_optimizer.param_groups:
            logger.info("Current learning rate for discriminators is: {}".format(param_group['lr']))

        if opt.warmup_value_null>epoch:
            w_kl = 0.0
            w_dis = 0.0
        else:
            w_kl = min(opt.w_kl, opt.w_kl*(epoch - opt.warmup_value_null) / (opt.warmup_value - opt.warmup_value_null))
            if epoch<opt.warmup_value_null_disc:
                w_dis = 0.0
            else:
                w_dis = opt.w_dis

        logger.info("Current w_kl {}".format(w_kl))
        logger.info("Current w_dis {}".format(w_dis))
        
        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            logger.info(phase)
            if phase == 'training':
                model.train()  # Set model to training mode
                for mod in opt.modalities:
                    netD[mod].train()
            else:
                model.eval()  # Set model to evaluate mode
                for mod in opt.modalities:
                    netD[mod].eval()

            running_loss = 0.0
            running_loss_img = 0.0
            running_kl = 0.0
            running_kl_multi = []
            running_loss_adv = 0.0
            running_loss_disc = 0.0
            epoch_samples = 0

            # Iterate over data
            for id_b in tqdm(ind_batch[phase]):
                batch = next(dataloaders[phase])
                name = batch[f"{first_mod}_name"][0]
                if phase=='training':
                    nw = opt.batch_size
                    w = np.random.randint(0, batch[first_mod].shape[-1]-nw-1)
                else:
                    w = 0
                    nw = batch[mod].shape[-1]
                
                nonempty_list = []
                imgs = dict()
                imgs_norm = dict()
                for mod in opt.modalities:
                    imgs[mod] = batch[mod].to(device).permute(0,4,1,2,3)
                    imgs[mod] = imgs[mod].reshape(-1, 1, *batch[mod].shape[2:4])[w:w+nw:,...]
                    if not torch.all(imgs[mod]==-1):
                        nonempty_list.append(mod)
                    imgs_norm[mod] = batch[f"{mod}_norm"].to(device).permute(0,4,1,2,3)
                    imgs_norm[mod] = imgs_norm[mod].reshape(-1, 1, *batch[f"{mod}_norm"].shape[2:4])[w:w+nw:,...]

                nb_voxels = np.prod(imgs[first_mod].shape)
                subset_mr = [k for k in nonempty_list if not 'us'==k]
                
                assert opt.modalities[0]=='us'
                total_subsets_mr = list(chain.from_iterable(combinations(subset_mr, r) for r in range(1,len(subset_mr)+1)))
                
                if epoch==1 and id_b==0 and phase=='training':
                    with torch.set_grad_enabled(False):
                        logger.info('[INFO] Display infos latent')
                        model({mod:imgs[first_mod]}, verbose=True)

                for subset in total_subsets_mr:
                    with torch.set_grad_enabled(phase == 'training'):
                        optimizer.zero_grad()

                        temp = 1.
                            
                        g_loss = 0.0
                        loss_img = 0.0
                        kl = 0.0
                        kl_multi = 0.0
                        outputs = dict()
                        
                        # Compute for for all modalities
                        output_img_complete, kls  = model({mod:imgs[mod].clone() for mod in nonempty_list}, temp)
                        kl += torch.sum(torch.stack(kls['prior']))
                        loss_img += l1_loss(output_img_complete, imgs_norm, nonempty_list, opt.l1_smooth)
                        listToStr = '_'.join([str(elem) for elem in nonempty_list])
                        outputs[listToStr] = output_img_complete
                        
                        # Compute for US
                        output_img_us, kls  = model({'us':imgs['us'].clone()}, temp)
                        kl += torch.sum(torch.stack(kls['prior']))
                        loss_img += l1_loss(output_img_us, imgs_norm, nonempty_list, opt.l1_smooth)
                        if w_dis>0:
                            g_loss += temp*gan(netD, output_img_us,  True)
                        outputs['us'] = output_img_us
                        
                        # Compute for MR subset
                        output_img_mr, kls  = model({mod:imgs[mod].clone() for mod in subset}, temp)
                        kl += torch.sum(torch.stack(kls['prior']))
                        loss_img += l1_loss(output_img_mr, imgs_norm, nonempty_list, opt.l1_smooth)
                        if w_dis>0:
                            g_loss += temp*gan(netD, output_img_mr,  True)
                        listToStr = '_'.join([str(elem) for elem in subset])
                        outputs[listToStr] = output_img_mr
                        
                        # Adds all losses                  
                        loss = loss_img + w_kl*kl/(nb_voxels*len(opt.modalities)) + w_dis*g_loss

                        # backward + optimize only if in training phase
                        if phase == 'training':
                            if loss>0:
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                

                    if epoch==opt.warmup_value_disc-1 and phase=='training':
                        for mod in nonempty_list:
                            real_pool[mod].add(imgs_norm[mod].contiguous().detach().data)

                    # Train Discriminator
                    if phase == 'training' and epoch>=opt.warmup_value_disc:
                        d_optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'training'):
                            d_loss = 0.0
                            # Positive
                            gt_d = dict()
                            for mod in opt.modalities:
                                if mod in nonempty_list:
                                    gt_d[mod] = imgs_norm[mod].contiguous().detach()
                                    real_pool[mod].add(gt_d[mod].data)
                                else:
                                    gt_d[mod] = real_pool[mod].query()
                            d_loss += gan(netD, gt_d, True, subset=nonempty_list)
                            
                            # Negatives
                            pred_d = {mod:torch.cat([v[mod].detach() for k,v in outputs.items() if not mod in k],0) for mod in opt.modalities}
                            d_loss += gan(netD, pred_d, False, subset=nonempty_list)

                            optimizer.zero_grad() 
                            d_loss.backward()
                            d_optimizer.step()
                    else:
                        d_loss = 0.0
                        
                                    
                    # statistics
                    epoch_samples += 1
                    running_loss += loss
                    running_loss_img += loss_img
                    running_loss_adv += g_loss
                    running_kl += kl
                    if kl_multi>0:
                        running_kl_multi.append(kl_multi)
                    running_loss_disc += d_loss

            running_loss = running_loss.item()
            if phase=='training':
                lr_scheduler.step()

            logger.info('{}  Loss Total: {:.4f}'.format(
                phase, running_loss / epoch_samples))

            logger.info('{}  Loss Img: {:.4f}'.format(
                phase, running_loss_img / epoch_samples))

            logger.info('{}  Loss KL: {:.4f}'.format(
                phase, running_kl / epoch_samples))
            
            logger.info('{}  Loss KL multi: {:.4f}'.format(
                phase, sum(running_kl_multi)/max(1,len(running_kl_multi))))

            logger.info('{}  Loss Adversarial: {:.4f}'.format(
                phase, running_loss_adv / epoch_samples))

            logger.info('{}  Loss Discriminator: {:.4f}'.format(
                phase, running_loss_disc / epoch_samples))

            epoch_loss = running_loss / epoch_samples

            if phase=='validation':
                if val_eval_criterion_MA is None: # first iteration
                    val_eval_criterion_MA = epoch_loss
                    best_val_eval_criterion_MA = val_eval_criterion_MA

                else: #update criterion
                    val_eval_criterion_MA = val_eval_criterion_alpha * val_eval_criterion_MA + (
                                1 - val_eval_criterion_alpha) * epoch_loss

                if val_eval_criterion_MA < best_val_eval_criterion_MA:
                    best_val_eval_criterion_MA = val_eval_criterion_MA
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format('main','best'))
            
                df = df._append({'epoch':epoch,
                    'best_epoch':best_epoch,
                    'MA':val_eval_criterion_MA,
                    'best_MA':best_val_eval_criterion_MA,  
                    'lr':param_group['lr']}, ignore_index=True)
                df.to_csv(df_path, index=False)
                if epoch==opt.warmup_value_disc - 10:
                    torch.save(model.state_dict(), save_path.format('main', epoch))
                if epoch==opt.stop_epochs:
                    continue_training=False
                    torch.save(model.state_dict(), save_path.format('main', epoch))
                    if epoch>=opt.warmup_value_disc:
                        for mod in opt.modalities:
                            torch.save(netD[mod].state_dict(), save_path.format(mod, epoch))
                    torch.save(model.state_dict(), save_path.format('main', 'final'))

        if (epoch%100==0 and epoch>0) or epoch==2:
            with torch.no_grad():

                torch.save(model.state_dict(), save_path.format('main', epoch))
                if epoch>=opt.warmup_value_disc:
                    for mod in opt.modalities:
                        torch.save(netD[mod].state_dict(), save_path.format(mod, epoch))
                    
                affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
                name = batch[f"{first_mod}_name"][0].replace(first_mod,'')+'{}_{}.nii.gz'
                saving_path = os.path.join(opt.model_dir, 'samples')
                
                # Save full    
                pred, _, feat  = model({mod:imgs[mod].clone() for mod in nonempty_list}, return_feat=True, return_cat=True)
                save_training(pred, affine, os.path.join(saving_path, name.format('full',epoch)))
                
                # Save us
                for temp in [0.1, 0.5, 1]:
                    # Save us
                    pred, _, feat  = model({'us':imgs['us'].clone()}, temp, return_feat=True, return_cat=True)
                    save_training(pred, affine, os.path.join(saving_path, name.format('us',f"{epoch}_{temp}")))
                    
                    # Save MR
                    for subset in total_subsets_mr:
                        pred, _, feat  = model({mod:imgs[mod].clone() for mod in subset}, temp, return_feat=True, return_cat=True)
                        listToStr = '_'.join([str(elem) for elem in subset])
                        save_training(pred, affine, os.path.join(saving_path,name.format(listToStr, f"{epoch}_{temp}")))
                    
                # Save sample
                # pred = model.sample(batch_size=imgs[first_mod].size(0), return_cat=True)
                # save_training(pred, affine, os.path.join(saving_path,name.format('sample', epoch)))

    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best epoch is {}'.format(best_epoch))

def main():
    opt = parsing_data()

    if opt.warmup_value<0:
        opt.warmup_value = opt.epochs

    # FOLDERS
    fold_dir = opt.model_dir
    fold_dir_model = os.path.join(fold_dir,'models')
    if not os.path.exists(fold_dir_model):
        os.makedirs(fold_dir_model)
    
    output_path = os.path.join(fold_dir,'samples')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger = create_logger(fold_dir)
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Initial lr: {opt.ini_lr}")
    logger.info(f"Total number of epochs: {opt.epochs}")
    logger.info(f"Number of modalities: {len(opt.modalities)}")
    logger.info(f"With residual: {opt.no_res}")
    logger.info(f"With se: {opt.no_se}")
    logger.info(f"Smooth l1: {opt.l1_smooth}")
    logger.info(f"Normalization type: {opt.type_normalization}")
    
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
    output_feat = len(opt.modalities)
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
        
    discriminators = {mod: NLayerDiscriminator(input_nc=1, n_class=1, ndf=64, n_layers=6).to(device) for mod in opt.modalities}

    logger.info("[INFO] Training")
    
    train(paths_dict, 
        model, 
        discriminators, 
        device, 
        logger,
        opt)



def parsing_data():
    parser = argparse.ArgumentParser(
        description='Training using MMHVAE')


    parser.add_argument('--model_dir',
                        type=str,
                        help='Save model directory')

    parser.add_argument('--dataset_split',
                        type=str,
                        default='./splits/split.csv',
                        help='Split file (.csv)')

    parser.add_argument('--path_data',
                        type=str,
                        default='../data/TPAMI/unnorm/',
                        help='Path to the unnorm dataset')

    parser.add_argument('--path_data_norm',
                        type=str,
                        default='../data/TPAMI/norm/',
                        help='Path to the norm dataset')
    
    parser.add_argument('--type_normalization',
                        type=str,
                        default='min-max',
                        help='Type of normalization')
    
    parser.add_argument('--ini_lr',
                    type=float,
                    default=2e-3,
                    help='Initial Learning Rate')

    parser.add_argument('--w_kl',
                    type=float,
                    default=0.001,
                    help='Weight KL Divergence')
    

    parser.add_argument('--w_dis',
                    type=float,
                    default=0.05,
                    help='Weight Discriminator')

    parser.add_argument('--seed',
                    type=int,
                    default=3)

    parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help='Batch Size (2D images)')

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

    parser.add_argument('--epochs',
                    type=int,
                    default=1000,
                    help='Number of training epochs')

    parser.add_argument('--stop_epochs',
                    type=int,
                    default=1000,
                    help='Stop training epochs')

    parser.add_argument('--warmup_value',
                    type=int,
                    default=50,
                    help='Warmup phase: number of epochs: final ramp epoch')

    parser.add_argument('--warmup_value_null',
                    type=int,
                    default=0,
                    help='Warmup phase KL+discriminator: start ramp epoch')
    
    parser.add_argument('--warmup_value_null_disc',
                    type=int,
                    default=800,
                    help='Warmup phase discriminator: final ramp epoch')

    parser.add_argument('--warmup_value_disc',
                    type=int,
                    default=790,
                    help='Warmup phase discriminator: start ramp epoch')

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
                    default=['us', 't2', 'cet1', 'flair'])
    
    
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

    parser.add_argument('--l1_smooth',
                        action='store_true', 
                        help='Smooth l1')
        
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()


