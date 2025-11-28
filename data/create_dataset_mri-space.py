## ==============================================
##  Goal: Generate a co-registered MRI / pre-dura iUS dataset
##  Reference space: reference MRI (high-res MRI space)
## ==============================================


import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

import os
from utils.coregistration import *


path_folder = './data/nrrd'
output_path_reg_us = './data/registration/affine'
output_path_reg_mr = './data/registration/rigid'

path_imgs = pd.read_csv('data/refs_files.csv', index_col=0).T
target_tumor = pd.read_csv('data/choice_target.csv', index_col=0).T

cases_toregister = path_imgs.columns.tolist()

print(f"Number of cases to register {len(cases_toregister)}")

output_path_final = './data/coregistered/mri-space'
os.makedirs(output_path_final, exist_ok=True)

errors = []
for case in tqdm(cases_toregister):
    try:
        # Define folder paths for case
        path_premr = os.path.join(path_folder, case, 'Preop-MR')
        path_intraus = os.path.join(path_folder, case, 'Intraop-US')
        path_annotations = os.path.join(path_folder, case, 'Annotations')
        
        # Define ref images for case
        img_path_mr = os.path.join(path_premr, path_imgs[case]['mr'])
        mr_flnm_or = path_imgs[case]['mr'].replace('.nrrd', '') 
        img_path_us = os.path.join(path_intraus, path_imgs[case]['us'])
        us_flnm_or = path_imgs[case]['us'].replace('.nrrd', '') 
        
        # Load ref images
        img_mr_ref = sitk.ReadImage(img_path_mr)
        img_us = sitk.ReadImage(img_path_us)

        # Load the transformation to align ref MRI and iUS
        res_filnm = os.path.join(output_path_reg_us, case, f"{case}-{case}-intraop-US-pre_dura-to-{mr_flnm_or}.tfm")
        transform_initial_inv = sitk.ReadTransform(res_filnm) # from iUS to ref MRI
        transform_initial = transform_initial_inv.GetInverse() # from ref MRI to iUS

        # Select MRIs to register
        T2s = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 't2' in k.lower() and not 'flair' in k.lower() and 'space' in k.lower()]
        flairs = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'flair' in k.lower() and '3d' in k.lower()]
        T1ce = [os.path.join(path_premr,k) for k in os.listdir(path_premr) if '.nrrd' in k if 'postcontrast' in k.lower() and '3d' in k.lower()]

        T2s = select_highest(T2s)
        flairs = select_highest(flairs)
        T1ce = select_highest(T1ce)
        
        T2s = [('t2',k) for k in T2s]
        flairs = [('flair',k) for k in flairs]
        T1ce = [('cet1',k) for k in T1ce]
        
        mrs_to_register = T2s + T1ce + flairs
                
        
        if len(mrs_to_register)>0:
            # 1. Resample all the MRI in the correct space
            for mod,mr_to_register in mrs_to_register:
                mr_to_register_flnm = os.path.basename(mr_to_register).replace('.nrrd', '')
                if mr_flnm_or in mr_to_register:
                    mr_transformation = sitk.Transform()
                else:
                    res_filnm = os.path.join(output_path_reg_mr, case, f"{case}-{mr_flnm_or}-to-{mr_to_register_flnm}.tfm")
                    mr_transformation = sitk.ReadTransform(res_filnm).GetInverse() # from MRI to ref MRI
                
                # We obtain the transformation for the MR to register
                final_transform = sitk.CompositeTransform([mr_transformation])
                
                # Now, we resample the MR to register
                img_mr = sitk.ReadImage(mr_to_register)
                img_mr = zero_mean(img_mr)
                img_mr = sitk.Cast(img_mr, sitk.sitkFloat32)
                img_mr = sitk.Resample(
                    img_mr,
                    img_mr_ref,
                    transform=final_transform,
                    interpolator=sitk.sitkLinear
                )

                # Save
                output_name = f"{case}-{mod}.nii.gz" 
                os.makedirs(os.path.join(output_path_final, case), exist_ok=True)
                sitk.WriteImage(
                    image=img_mr, 
                    fileName=os.path.join(output_path_final, case, output_name),
                    useCompression=True
                )
        
        # 2. Resample and save the iUS in the correct space   
        img_us_reg = sitk.Resample(
                    img_us,
                    img_mr_ref,
                    transform_initial_inv,
                    sitk.sitkLinear
        )
        
        sitk.WriteImage(
            image=img_us_reg, 
            fileName=os.path.join(output_path_final, case, f"{case}-us.nii.gz"),
            useCompression=True
        )
        
        # 3. Resample the segmentation of the target in the correct space 
        seg_flnm = target_tumor[case]['target']
        mr_ref_label = seg_flnm.replace('SEG-tumor_target-','').replace('SEG-tumor-', '')
        if mr_flnm_or in mr_ref_label:
            transform_label = sitk.Transform()
        else:
            transform_label_filnm = os.path.join(output_path_reg_mr, case, f"{case}-{mr_flnm_or}-to-{mr_ref_label}.tfm")
            transform_label = sitk.ReadTransform(transform_label_filnm).GetInverse()
        
        # We obtain the transformation for the seg to register
        final_transform_label = transform_label
        
        # Now, we resample the segmentation
        path_seg = os.path.join(path_annotations, f"{seg_flnm}.nrrd")
        seg = sitk.ReadImage(path_seg)
        
        seg_resample = resample_seg(
                original_lab=seg,
                target=img_mr_ref,
                transformation=final_transform_label
        )
        
        sitk.WriteImage(
            image=seg_resample, 
            fileName=os.path.join(output_path_final, case, f"{case}-target.nii.gz"),
            useCompression=True
        )
        
    except Exception as e:
        print(f"error with {case} - {e}")
        errors.append(case)

if len(errors)>0:
    print(errors)
else:
    print('All good - the script worked on all the dataset!')


