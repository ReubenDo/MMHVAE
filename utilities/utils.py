import os
import logging
import pandas as pd
import torch
import nibabel as nib  

import random
import numpy as np

import types
import warnings

import torch
from torch.nn import init

_seed = None
_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
NP_MAX = np.iinfo(np.uint32).max
MAX_SEED = NP_MAX + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32


def create_logger(folder):
    """Create a logger to save logs."""
    compt = 0
    while os.path.exists(os.path.join(folder,f"logs_{compt}.txt")):
        compt+=1
    logname = os.path.join(folder,f"logs_{compt}.txt")
    
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(logname, mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    return logger 

def infinite_iterable(i):
    while True:
        yield from i

def poly_lr(epoch: int, max_epochs: int, initial_lr: float, min_lr: float=1e-5, exponent: float=0.9) -> float:
    return min_lr + (initial_lr - min_lr) * (1 - epoch / max_epochs)**exponent


def init_training_variables(model, netD, opt, logger, phase='training'):
    df_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = int(df.iloc[-1]['epoch']/10)*10
    else:
        epoch = 0
        
    if epoch>=10:
        best_epoch = epoch
        val_eval_criterion_MA = df.iloc[-1]['MA']
        best_val_eval_criterion_MA = df.iloc[-1]['best_MA']
        initial_lr = df.iloc[-1]['lr']
        model.load_state_dict(torch.load(save_path.format('main',epoch)))
        logger.info(f"Loading model from {save_path.format('main',epoch)}")
        try:
            for mod in opt.modalities:
                netD[mod].load_state_dict(torch.load(save_path.format(mod,epoch)))
                logger.info(f"Loading {mod} distcriminator from {save_path.format(mod,epoch)}")
        except Exception as e:
            logger.info(e)
    else: # If training from scratch
        df = pd.DataFrame(columns=['epoch','best_epoch', 'MA', 'best_MA', 'lr'])
        val_eval_criterion_MA = None
        best_epoch = 0
        epoch = 0
        best_val_eval_criterion_MA = 1e10
        initial_lr = opt.ini_lr
        # if not opt.load_net is None:
        #     model.load_state_dict(torch.load(opt.load_net))
        #     logger.info(f"Loading model from {opt.load_net}")
            
        
    return df, val_eval_criterion_MA, best_val_eval_criterion_MA, best_epoch, epoch, initial_lr


def init_training_variables_singled(model, netD, opt, logger, phase='training'):
    df_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = int(df.iloc[-1]['epoch']/10)*10
    else:
        epoch = 0
        
    if epoch>=10:
        best_epoch = epoch
        val_eval_criterion_MA = df.iloc[-1]['MA']
        best_val_eval_criterion_MA = df.iloc[-1]['best_MA']
        initial_lr = df.iloc[-1]['lr']
        model.load_state_dict(torch.load(save_path.format('main',epoch)))
        logger.info(f"Loading model from {save_path.format('main',epoch)}")
        try:
            netD.load_state_dict(torch.load(save_path.format('discri',epoch)))
            logger.info(f"Loading distcriminator from {save_path.format('discri',epoch)}")
        except Exception as e:
            logger.info(e)
    else: # If training from scratch
        df = pd.DataFrame(columns=['epoch','best_epoch', 'MA', 'best_MA', 'lr'])
        val_eval_criterion_MA = None
        best_epoch = 0
        epoch = 0
        best_val_eval_criterion_MA = 1e10
        initial_lr = opt.ini_lr
        # if not opt.load_net is None:
        #     model.load_state_dict(torch.load(opt.load_net))
        #     logger.info(f"Loading model from {opt.load_net}")
            
        
    return df, val_eval_criterion_MA, best_val_eval_criterion_MA, best_epoch, epoch, initial_lr

def save(pred, affine, path):
    pred = (255*(pred+1)/2).int()
    pred = pred.permute(2,3,0,1).squeeze().cpu().numpy()
    nib.Nifti1Image(pred, affine).to_filename(path)

def save_training(pred, affine, path):
    new_pred = []
    for i in range(pred.shape[1]):
        img_data = pred[:,i,...]
        mask = img_data>-1
        max_data = img_data[mask].max()
        min_data = img_data[mask].min()
        sub = min_data
        div = max(256/255*(max_data-min_data),1e-8)
        img_data = (img_data - sub) /div +1/256
        img_data[~mask] = 0
        img_data = (255*img_data).int()
        new_pred.append(img_data)
        
    new_pred = torch.stack(new_pred,1)
    new_pred = new_pred.permute(2,3,0,1).squeeze().cpu().numpy()
    nib.Nifti1Image(new_pred, affine).to_filename(path)
    

def quantisize(img_data, levels=256, lower=0.0, upper=99.95):
    min_data = np.percentile(img_data, lower)
    max_data = np.percentile(img_data, upper)
    
    img_data[img_data<min_data] = min_data 
    img_data[img_data>max_data] = max_data
    img_data = (img_data-min_data) / (max_data - min_data+1e-8)
    img_data = np.digitize(img_data.squeeze(), np.arange(0,levels-1)/(levels-1)  ) 

    return img_data.astype(np.uint8)    

    
def save_feature(features, paths, affine):
    pred = np.stack([pred.permute(2,3,0,1).squeeze().cpu().numpy() for pred in features],0)
    out = []
    nb_channels = pred.shape[-1]
    mask = ~np.all(pred[0,...]==0,-1, keepdims=True)
    nb_imgs = pred.shape[0]
    for c in range(nb_channels):
        out.append(quantisize(pred[...,c],levels=256,lower=1, upper=99.))
    out = np.stack(out,-1)
    for i_image in range(nb_imgs): 
        nib.Nifti1Image(mask*out[i_image,...], affine).to_filename(paths[i_image])
        
def save_feature_tpami(features, paths, affine):
    pred = np.stack([pred.permute(2,3,0,1).squeeze().cpu().numpy() for pred in features],0)
    out = []
    nb_channels = pred.shape[-1]
    mask = ~np.all(pred[0,...]==0,-1, keepdims=True)
    nb_imgs = pred.shape[0]
    for i_image in range(nb_imgs): 
        nib.Nifti1Image(mask*pred[i_image,...], affine).to_filename(paths[i_image])
    


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
    
def test_nonempty(path_img):
    img_data = nib.load(path_img).get_fdata()
    if not np.all(img_data==0):
        return True
    else:
        return False
    
def set_determinism(
    seed: int | None = NP_MAX,
    use_deterministic_algorithms: bool | None = None,
   
) -> None:
    """
    Set random seed for modules to enable or disable deterministic training.

    Args:
        seed: the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        use_deterministic_algorithms: Set whether PyTorch operations must use "deterministic" algorithms.
        additional_settings: additional settings that need to set random seed.

    Note:

        This function will not affect the randomizable objects in :py:class:`monai.transforms.Randomizable`, which
        have independent random states. For those objects, the ``set_random_state()`` method should be used to
        ensure the deterministic behavior (alternatively, :py:class:`monai.data.DataLoader` by default sets the seeds
        according to the global random state, please see also: :py:class:`monai.data.utils.worker_init_fn` and
        :py:class:`monai.data.utils.set_rnd`).
    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)


    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(use_deterministic_algorithms)
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")