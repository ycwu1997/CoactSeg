import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk

def ImageResample(sitk_image, new_spacing = [0.8,0.8,0.8], is_label = False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]
    if not is_label:
        print("original shape:", size)
        print("resampled shape:", new_size)
        print("original spacing:", spacing)
        print("resampled spacing:", new_spacing_refine)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    newimage = resample.Execute(sitk_image)
    return newimage

def crop_roi(image, mask):
    ### crop based on brain segmentation
    w, h, d = mask.shape
    tempL = np.nonzero(mask)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])
    
    minx = max(minx - 15, 0)
    maxx = min(maxx + 15, w)
    miny = max(miny - 15, 0)
    maxy = min(maxy + 15, h)
    minz = max(minz - 15, 0)
    maxz = min(maxz + 15, d)
    image = image * mask
    image = image[minx:maxx, miny:maxy, minz:maxz].astype(np.float32)
    return image

listt = sorted(glob('./training/*/*.nii'))

for item in tqdm(listt):
    name = str(item)
    name_id = name[11:22]
    image_name = name
    image_mask_name = "./brain/" + name[-36:-4] + "_mask.nii.gz"
    gt_name = name[:-10] + "_mask.nii.gz"
    print("data id:", name_id)

    itk_label = sitk.ReadImage(gt_name)
    itk_label = ImageResample(itk_label, is_label = True)
    label = sitk.GetArrayFromImage(itk_label)


    itk_img = sitk.ReadImage(image_name)
    origin = itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()
    itk_img = ImageResample(itk_img)
    image = sitk.GetArrayFromImage(itk_img)
    
    itk_img = sitk.ReadImage(image_mask_name)
    itk_img = ImageResample(itk_img, is_label = True)
    image_mask = sitk.GetArrayFromImage(itk_img)

    assert(np.shape(image)==np.shape(image_mask))
    image = crop_roi(image, image_mask)
    image = (image - np.mean(image)) / np.std(image)

    image_cropped = sitk.GetImageFromArray(image)
    image_cropped.SetOrigin(origin)
    image_cropped.SetDirection(direction)
    image_cropped.SetSpacing(space)
    sitk.WriteImage(image_cropped, "./images/"+name_id+".nii.gz")


    label = crop_roi(label, image_mask)
    print("sum_label:%d" % np.sum(label))
    print("cropped shape:", label.shape)
    image_cropped = sitk.GetImageFromArray(label)
    image_cropped.SetOrigin(origin)
    image_cropped.SetDirection(direction)
    image_cropped.SetSpacing(space)
    sitk.WriteImage(image_cropped, "./labels/"+name_id+".nii.gz")


    f = h5py.File(('./data_h5/'+name_id + '_norm.h5'), 'w')
    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()
