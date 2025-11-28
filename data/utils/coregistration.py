import SimpleITK as sitk
import numpy as np
from PIL import Image
import torchio as tio

def reshape(img, size=(192,192,192)):
    subject = tio.Subject(
    us=tio.ScalarImage.from_sitk(img),
    )
    transform = tio.CropOrPad(size)
    transformed = transform(subject)
    return transformed['us'].as_sitk()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def clean_cmdline(cmd):
    return cmd.replace(' ', '\ ').replace('(','\(').replace(')','\)')

FLIPXY_44 = np.diag([-1, -1, 1, 1])
def _to_itk_convention(matrix):
    """RAS to LPS"""
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.linalg.inv(matrix)
    return matrix

def _matrix_to_itk_transform(matrix, dimensions=3):
    matrix = _to_itk_convention(matrix)
    rotation = matrix[:dimensions, :dimensions].ravel().tolist()
    translation = matrix[:dimensions, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform


def _write_itk_matrix(matrix, tfm_path):
    """The tfm file contains the matrix from floating to reference."""
    transform = _matrix_to_itk_transform(matrix).GetInverse()
    transform.WriteTransform(str(tfm_path))

def get_spacing(spacing, spacing_target, thr_low=0.2, name=None):
    spacing = [float(k) for k in spacing]
    if abs(spacing[0]-spacing[1])/spacing[0] < thr_low:
        if abs(spacing[0]-spacing[2])/spacing[0] < thr_low:
            return (spacing_target, spacing_target, spacing_target)
        else:
            return (spacing_target,spacing_target,spacing[2])
        
    elif abs(spacing[2]-spacing[1])/spacing[0] < thr_low:
        if abs(spacing[0]-spacing[2])/spacing[0] < thr_low:
            return (spacing_target, spacing_target, spacing_target)
        else:
            return (spacing_target,spacing_target,spacing[0])
        
    elif abs(spacing[0]-spacing[2])/spacing[0] < thr_low:
        if abs(spacing[1]-spacing[0])/spacing[0] < thr_low:
            return (spacing_target, spacing_target, spacing_target)
        else:
            return (spacing_target,spacing_target,spacing[1])
    else:
        return 0 / 0
    
def resample(img,new_spacing,interpolator=sitk.sitkLinear):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    img_resample = sitk.Resample(img, new_size, sitk.Transform(), interpolator,
                            img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                            img.GetPixelID())
    return img_resample

def _find_zeros(img_array):
    if len(img_array.shape) == 4:
        img_array = np.amax(img_array, axis=3)
    assert len(img_array.shape) == 3
    x_dim, y_dim, z_dim = tuple(img_array.shape)
    x_zeros, y_zeros, z_zeros = np.where(img_array == 0.)
    # x-plans that are not uniformly equal to zeros
    
    try:
        x_to_keep, = np.where(np.bincount(x_zeros) < y_dim * z_dim)
        x_min = min(x_to_keep) 
        x_max = max(x_to_keep) + 1
    except Exception :
        x_min = 0
        x_max = x_dim
    try:
        y_to_keep, = np.where(np.bincount(y_zeros) < x_dim * z_dim)
        y_min = min(y_to_keep) 
        y_max = max(y_to_keep) + 1
    except Exception :
        y_min = 0
        y_max = y_dim
    try :
        z_to_keep, = np.where(np.bincount(z_zeros) < x_dim * y_dim)
        z_min = min(z_to_keep) 
        z_max = max(z_to_keep) + 1
    except:
        z_min = 0
        z_max = z_dim
    return x_min, x_max, y_min, y_max, z_min, z_max

def crop_with_ref(img, ref):
    ref_data =  sitk.GetArrayFromImage(ref).transpose().squeeze()
    x_min, x_max, y_min, y_max, z_min, z_max = _find_zeros(ref_data)

    x_max = ref_data.shape[0] - x_max
    y_max = ref_data.shape[1] - y_max
    z_max = ref_data.shape[2] - z_max
    bounds_parameters = [x_min, x_max, y_min, y_max, z_min, z_max]
    low = bounds_parameters[::2]
    high = bounds_parameters[1::2]
    low = [int(k) for k in low]
    high = [int(k) for k in high]

    output = sitk.Crop(img, low, high)
    return output 
    
def quantisize(img, levels=256, lower=0.0, upper=99.95):
    img_data = sitk.GetArrayFromImage(img).astype(np.float32)
    mask = img_data>0
    min_data = np.percentile(img_data[mask], lower)
    max_data = np.percentile(img_data[mask], upper)
    
    img_data[~mask] = min_data - 1
    img_data[img_data>max_data] = max_data
    img_data = (img_data-min_data) / (max_data - min_data+1e-8)
    img_data = np.digitize(img_data.squeeze(), np.arange(0,levels-1)/(levels-1)  ) 
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(img)
    output =  sitk.Cast(output, sitk.sitkUInt8)
    return output

def mask(img_to_mask, ref):
    img_data = sitk.GetArrayFromImage(img_to_mask).astype(np.float32)
    ref_data = sitk.GetArrayFromImage(ref).astype(np.float32)
    img_data*= (ref_data>0).astype(np.float32)
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(img_to_mask)
    output =  sitk.Cast(output, sitk.sitkFloat32)
    return output   

def zeros_like(ref):
    img_data = np.zeros_like(sitk.GetArrayFromImage(ref).astype(np.uint8))
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(ref)
    output =  sitk.Cast(output, sitk.sitkFloat32)
    return output

    
def get_mask(ref):
    ref_data = sitk.GetArrayFromImage(ref).astype(np.float32)
    img_data = (ref_data>0).astype(np.float32)
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(ref)
    output =  sitk.Cast(output, sitk.sitkFloat32)
    return output 

def select_highest(list_imgs):
    if len(list_imgs)>0:
        res_setscans = {k: np.mean(sitk.ReadImage(k).GetSpacing()) for k in list_imgs } 
        return  [min(res_setscans, key=res_setscans.get)]
    else:
        return []
    
    
def crop(img):
    img_data =  sitk.GetArrayFromImage(img).transpose().squeeze()
    x_min, x_max, y_min, y_max, z_min, z_max = find_zeros(img_data)

    x_max = img_data.shape[0] - x_max
    y_max = img_data.shape[1] - y_max
    z_max = img_data.shape[2] - z_max
    bounds_parameters = [x_min, x_max, y_min, y_max, z_min, z_max]
    low = bounds_parameters[::2]
    high = bounds_parameters[1::2]
    low = [int(k) for k in low]
    high = [int(k) for k in high]
    output = sitk.Crop(img, low, high)
    return output 

def find_zeros(img_array):
    if len(img_array.shape) == 4:
        img_array = np.amax(img_array, axis=3)
    assert len(img_array.shape) == 3
    x_dim, y_dim, z_dim = tuple(img_array.shape)
    x_zeros, y_zeros, z_zeros = np.where(img_array == 0.)
    # x-plans that are not uniformly equal to zeros
    
    try:
        x_to_keep, = np.where(np.bincount(x_zeros) < y_dim * z_dim)
        x_min = min(x_to_keep) 
        x_max = max(x_to_keep) + 1
    except Exception :
        x_min = 0
        x_max = x_dim
    try:
        y_to_keep, = np.where(np.bincount(y_zeros) < x_dim * z_dim)
        y_min = min(y_to_keep) 
        y_max = max(y_to_keep) + 1
    except Exception :
        y_min = 0
        y_max = y_dim
    try :
        z_to_keep, = np.where(np.bincount(z_zeros) < x_dim * y_dim)
        z_min = min(z_to_keep) 
        z_max = max(z_to_keep) + 1
    except:
        z_min = 0
        z_max = z_dim
    return x_min, x_max, y_min, y_max, z_min, z_max

def get_spacing(spacing, spacing_target, thr_low=0.2, name=None):
    spacing = [float(k) for k in spacing]
    if abs(spacing[0]-spacing[1])/spacing[0] < thr_low:
        if abs(spacing[0]-spacing[2])/spacing[0] < thr_low:
            return (spacing_target, spacing_target, spacing_target)
        else:
            return (spacing_target,spacing_target,spacing[2])
        
    elif abs(spacing[2]-spacing[1])/spacing[0] < thr_low:
        if abs(spacing[0]-spacing[2])/spacing[0] < thr_low:
            return (spacing_target, spacing_target, spacing_target)
        else:
            return (spacing_target,spacing_target,spacing[0])
        
    elif abs(spacing[0]-spacing[2])/spacing[0] < thr_low:
        if abs(spacing[1]-spacing[0])/spacing[0] < thr_low:
            return (spacing_target, spacing_target, spacing_target)
        else:
            return (spacing_target,spacing_target,spacing[1])
    else:
        return 0 / 0

def resample(img,new_spacing,interpolator=sitk.sitkLinear):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    img_resample = sitk.Resample(img, new_size, sitk.Transform(), interpolator,
                            img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                            img.GetPixelID())
    return img_resample

import torchio as tio
def reshape(img, size=(192,192,160)):
    img_crop = crop(img)
    z_size = img_crop.GetSize()[-1]
    subject = tio.Subject(
    us=tio.ScalarImage.from_sitk(img_crop), 
    )
    transform = tio.CropOrPad((size[0], size[1], z_size))
    transformed = transform(subject)
    return transformed['us'].as_sitk()


def zero_mean(img):
    img_data = sitk.GetArrayFromImage(img).astype(np.float32)
    img_data = img_data - img_data.min()
    output = sitk.GetImageFromArray(img_data)
    output.CopyInformation(img)
    output =  sitk.Cast(output, sitk.sitkInt16)
    return output


def resample_seg(original_lab, target, transformation, labels=None):
    arrays = []
    labels = np.unique(sitk.GetArrayFromImage(original_lab).astype(np.uint8)).tolist()
    for i in labels:
        lab = sitk.GetImageFromArray((sitk.GetArrayFromImage(original_lab).astype(np.uint8)==i).astype(np.float32))
        lab.CopyInformation(original_lab)
        lab_resample = sitk.Resample(lab, target, transformation, sitk.sitkLinear)
        arrays.append(sitk.GetArrayFromImage(lab_resample))

    final_seg = np.argmax(np.stack(arrays,0),0).astype(np.uint8)
    # final_seg = np.stack(arrays,-1)
    final_seg = sitk.GetImageFromArray(final_seg)
    final_seg.CopyInformation(lab_resample)
    return final_seg 