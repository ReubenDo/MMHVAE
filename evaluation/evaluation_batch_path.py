import SimpleITK as sitk
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm 
# import matplotlib.pyplot as plt
import lpips
import torch 
import time

import argparse

def arg_parse():
    parser = argparse.ArgumentParser(
            description="main.py")
    parser.add_argument(
        "-path", default=0, type=str, help="")
    args = parser.parse_args()
    return args

args = arg_parse()

fold = int(args.path.split('fold')[1][0])

lpips_loss = lpips.LPIPS(net='alex').cuda()

def lpips_vgg(array1, array2):
    with torch.no_grad():
        t1 = torch.from_numpy(array1)[None,None,...].repeat(1,3,1,1).float().cuda()
        t2 = torch.from_numpy(array2)[None,None,...].repeat(1,3,1,1).float().cuda()
        t1 = 2*t1/255.0 - 1
        t2 = 2*t2/255.0 - 1
        return lpips_loss(t1,t2).item()


scores = {'case':[], 'fold':[], 'method':[], 'input_mod':[], 'target_mod':[], 'slice':[], 'L1':[], 'SSIM':[], 'PSNR':[], 'LPIPS':[]}

modalities = ['us', 't2', 'cet1', 'flair']

path_method = args.path
assert os.path.isdir(path_method)
method = os.path.basename(os.path.normpath(path_method.replace(' ','')))
print(f"model {method} located at: {path_method}")
inputs = [k for k in os.listdir(path_method) if os.path.isdir(os.path.join(path_method, k))]
inference_path = os.path.join(path_method, 'inference_multitemp')

inference_path_gt = os.path.join(path_method, 'inference_multitemp')
files = sorted([k for k in os.listdir(inference_path) if '.nii' in k and not 'gt' in k])
for inde, f in tqdm(enumerate(files),total=len(files)):
    # if inde%10==0:
    #     print(f'Case {inde}/{len(files)}')
    path_f = os.path.join(inference_path,f)
    input_list = f.split('-')[1].split('.ni')[0]
    
    case = f.split('-')[0]
    output_all =  nib.load(path_f).get_fdata()
    gt_all = nib.load(os.path.join(inference_path_gt,f"{case}-gt_img.nii.gz")).get_fdata()
    for i in range(gt_all.shape[-1]):
        gt = gt_all[...,i]
        output = output_all[...,i]
        target_mod = modalities[i]
        if np.sum(gt)>0:
            for i in range(gt.shape[-1]):
                output_slice = output[...,i]
                gt_slice = gt[...,i]
                if np.sum(gt_slice)>0:
                    l1_score = np.abs(output_slice-gt_slice).mean()
                    psnr_score = psnr(gt_slice, output_slice, data_range=gt_slice.max() - gt_slice.min()) 
                    ssmi_score = ssim(gt_slice, output_slice, data_range=gt_slice.max() - gt_slice.min(), channel_axis=0)
                    lpips_score = lpips_vgg(gt_slice, output_slice)
                    
                    scores['case']+=[case]
                    scores['fold']+=[0]
                    scores['method']+=[method]
                    scores['input_mod']+=[input_list] 
                    scores['target_mod']+=[target_mod] 
                    scores['slice']+=[i]
                    scores['L1']+=[l1_score]
                    scores['SSIM']+=[ssmi_score]
                    scores['PSNR']+=[psnr_score]
                    scores['LPIPS']+=[lpips_score]
                    
                    
df = pd.DataFrame(scores)
df.to_csv(f'evaluation/test_fold{fold}_{method}.csv', index=False)
                
                
            
    