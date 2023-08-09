import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk

def ImageResample(sitk_image, new_spacing = [0.5,0.75,0.75], is_label = False):
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

listt = sorted(glob('./training/*'))

for item in tqdm(listt):
    name = str(item)
    name_id = name[-3:]
    first_point_name = name+"/flair_time01_on_middle_space.nii.gz"
    first_point_mask_name = name+"/flair_time01_on_middle_space_bet_mask.nii.gz"
    second_point_name = name+"/flair_time02_on_middle_space.nii.gz"
    second_point_mask_name = name+"/flair_time01_on_middle_space_bet_mask.nii.gz"
    gt_name = name+"/ground_truth.nii.gz"
    print("data id:", name_id)

    itk_label = sitk.ReadImage(gt_name)
    itk_label = ImageResample(itk_label, is_label = True)
    label = sitk.GetArrayFromImage(itk_label)

    itk_img = sitk.ReadImage(first_point_name)
    origin = itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()
    itk_img = ImageResample(itk_img)
    image_1 = sitk.GetArrayFromImage(itk_img)
    
    itk_img = sitk.ReadImage(first_point_mask_name)
    itk_img = ImageResample(itk_img, is_label = True)
    image_1_mask = sitk.GetArrayFromImage(itk_img)

    assert(np.shape(image_1)==np.shape(image_1_mask))
    image_1_point = crop_roi(image_1, image_1_mask)
    image_1_point = (image_1_point - np.mean(image_1_point)) / np.std(image_1_point)

    image_cropped = sitk.GetImageFromArray(image_1_point)
    image_cropped.SetOrigin(origin)
    image_cropped.SetDirection(direction)
    image_cropped.SetSpacing(space)
    sitk.WriteImage(image_cropped, "./images/"+name_id+"_first_point.nii.gz")

    itk_img = sitk.ReadImage(second_point_name)
    itk_img = ImageResample(itk_img)
    image_2 = sitk.GetArrayFromImage(itk_img)

    itk_img = sitk.ReadImage(second_point_mask_name)
    itk_img = ImageResample(itk_img, is_label = True)
    image_2_mask = sitk.GetArrayFromImage(itk_img)

    assert(np.shape(image_2)==np.shape(image_2_mask))
    image_2_point = crop_roi(image_2, image_2_mask)
    image_2_point = (image_2_point - np.mean(image_2_point)) / np.std(image_2_point)

    image_cropped = sitk.GetImageFromArray(image_2_point)
    image_cropped.SetOrigin(origin)
    image_cropped.SetDirection(direction)
    image_cropped.SetSpacing(space)
    sitk.WriteImage(image_cropped, "./images/"+name_id+"_second_point.nii.gz")

    label = crop_roi(label, image_1_mask)
    print("sum_label:%d" % np.sum(label))
    print("cropped shape:", label.shape)
    image_cropped = sitk.GetImageFromArray(label)
    image_cropped.SetOrigin(origin)
    image_cropped.SetDirection(direction)
    image_cropped.SetSpacing(space)
    sitk.WriteImage(image_cropped, "./labels/"+name_id+".nii.gz")


    f = h5py.File(('./h5/data'+name_id + '_norm.h5'), 'w')
    f.create_dataset('image_1', data=image_1_point, compression="gzip")
    f.create_dataset('image_2', data=image_2_point, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()

    image_cropped_save1 = sitk.GetImageFromArray(image_1_point)
    image_cropped_save1.SetOrigin(origin)
    image_cropped_save1.SetDirection(direction)
    image_cropped_save1.SetSpacing(space)
    sitk.WriteImage(image_cropped_save1, "./images/"+name_id+"_1.nii.gz")

    image_cropped_save2 = sitk.GetImageFromArray(image_2_point)
    image_cropped_save2.SetOrigin(origin)
    image_cropped_save2.SetDirection(direction)
    image_cropped_save2.SetSpacing(space)
    sitk.WriteImage(image_cropped_save2, "./images/"+name_id+"_2.nii.gz")

    label_cropped_save = sitk.GetImageFromArray(label)
    label_cropped_save.SetOrigin(origin)
    label_cropped_save.SetDirection(direction)
    label_cropped_save.SetSpacing(space)
    sitk.WriteImage(label_cropped_save, "./labels/"+name_id+".nii.gz")