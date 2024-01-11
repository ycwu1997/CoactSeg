import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import numpy as np

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
    with open('./data/val.list', 'r') as f:
            image_list = f.readlines()
    image_list = [item.replace('\n','') for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        if "MSSEG2" in image_path:
            h5f = h5py.File(image_path, 'r')
            image_1 = h5f['image_1'][:]
            image_2 = h5f['image_2'][:]
            label = h5f['label'][:]
            _, _, prediction_sub, _, _, _ = test_single_case_all(model, image_1, image_2, stride_xy, stride_z, patch_size, num_classes=num_classes)
            if np.sum(prediction_sub)==0:
                dice = 0
            else:
                dice = metric.binary.dc(prediction_sub, label)
            total_dice += dice
        else:
            h5f = h5py.File(image_path, 'r')
            image_1 = h5f['image'][:]
            image_2 = h5f['image'][:]
            label = h5f['label'][:]
            prediction_1, _, _, _, _, _ = test_single_case_all(model, image_1, image_2, stride_xy, stride_z, patch_size, num_classes=num_classes)
            if np.sum(prediction_1)==0:
                dice = 0
            else:
                dice = metric.binary.dc(prediction_1, label)
            total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric1 = 0.0
    total_metric2 = 0.0
    for image_path in loader:
        if "MSSEG2" in image_path:
            h5f = h5py.File(image_path, 'r')
            image_1 = h5f['image_1'][:]
            image_2 = h5f['image_2'][:]
            label = h5f['label'][:]
            if preproc_fn is not None:
                image = preproc_fn(image)
            prediction_1, _, prediction_sub, _, _, _ = test_single_case_all(model, image_1, image_2, stride_xy, stride_z, patch_size, num_classes=num_classes)
            _, prediction_2, _, _, _, _ = test_single_case_all(model, image_2, image_2, stride_xy, stride_z, patch_size, num_classes=num_classes)
            prediction = prediction_sub
            if nms:
                    prediction = getLargestCC(prediction)
            if np.sum(prediction)==0:
                single_metric = (0,0,0,0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])
            total_metric1 += np.asarray(single_metric)
        else:
            h5f = h5py.File(image_path, 'r')
            image_1 = h5f['image'][:]
            image_2 = h5f['image'][:]
            label = h5f['label'][:]
            if preproc_fn is not None:
                image = preproc_fn(image)
            prediction_1, prediction_2, prediction_sub, score_map_1, score_map_2, score_map_sub = test_single_case_all(model, image_1, image_2, stride_xy, stride_z, patch_size, num_classes=num_classes)
            prediction = prediction_1
            if nms:
                    prediction = getLargestCC(prediction)
            if np.sum(prediction)==0:
                single_metric = (0,0,0,0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])
            total_metric2 += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction_1.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_1.nii.gz" % ith)
            nib.save(nib.Nifti1Image(prediction_2.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_2.nii.gz" % ith)
            nib.save(nib.Nifti1Image(prediction_sub.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_sub.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image_1[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_img_1.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image_2[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_img_2.nii.gz" % ith)
            nib.save(nib.Nifti1Image((image_2-image_1).astype(np.float32), np.eye(4)), test_save_path +  "%02d_img_sub.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_gt.nii.gz" % ith)

        ith += 1

    avg_metric1 = total_metric1 / 8
    avg_metric2 = total_metric2 / 8
    print('average metric_public_(dice, jc, hd, asd, precision, se, sp, F1): {}'.format(avg_metric1))
    print('average metric_inhouse_(dice, jc, hd, asd, precision, se, sp, F1): {}'.format(avg_metric2))
    with open(test_save_path+'../{}_performance.txt'.format(model_name), 'w') as f:
        f.writelines('average metric_public_(dice, jc, hd, asd, precision, se, sp, F1): {}'.format(avg_metric1))
        f.writelines('average metric_inhouse_(dice, jc, hd, asd, precision, se, sp, F1): {}'.format(avg_metric2))
    return avg_metric1

def test_single_case_all(model, image_1, image_2, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image_1.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image_1 = np.pad(image_1, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image_2 = np.pad(image_2, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image_1.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map_1 = np.zeros((num_classes, ) + image_1.shape).astype(np.float32)
    score_map_2 = np.zeros((num_classes, ) + image_1.shape).astype(np.float32)
    score_map_sub = np.zeros((num_classes, ) + image_1.shape).astype(np.float32)
    cnt = np.zeros(image_1.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch_1 = image_1[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_1 = np.expand_dims(np.expand_dims(test_patch_1,axis=0),axis=0).astype(np.float32)
                test_patch_1 = torch.from_numpy(test_patch_1).cuda()
                test_patch_2 = image_2[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_2 = np.expand_dims(np.expand_dims(test_patch_2,axis=0),axis=0).astype(np.float32)
                test_patch_2 = torch.from_numpy(test_patch_2).cuda()
                test_sub = test_patch_2-test_patch_1
                test_patch = torch.cat([test_patch_1, test_patch_2, test_sub], dim=1)
                
                with torch.no_grad():
                    y1, y2, y3 = model(test_patch)
                    y1, y2, y3 = F.softmax(y1, dim=1), F.softmax(y2, dim=1), F.softmax(y3, dim=1)
                y1, y2, y3 = y1.cpu().data.numpy(), y2.cpu().data.numpy(), y3.cpu().data.numpy()
                y1, y2, y3 = y1[0,1,:,:,:], y2[0,1,:,:,:], y3[0,1,:,:,:]
                score_map_1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y1
                score_map_2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y2
                score_map_sub[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_sub[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y3
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map_1 = score_map_1/np.expand_dims(cnt,axis=0)
    score_map_2 = score_map_2/np.expand_dims(cnt,axis=0)
    score_map_sub = score_map_sub/np.expand_dims(cnt,axis=0)

    label_map_1 = (score_map_1[0]>0.5).astype(np.int)
    label_map_2 = (score_map_2[0]>0.5).astype(np.int)
    label_map_sub = (score_map_sub[0]>0.5).astype(np.int)
    if add_pad:
        label_map_1 = label_map_1[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        label_map_2 = label_map_2[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        label_map_sub = label_map_sub[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_1 = score_map_1[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_2 = score_map_2[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_sub = score_map_sub[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]

    return label_map_1, label_map_2, label_map_sub, score_map_1, score_map_2, score_map_sub


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    precision = metric.binary.precision(pred, gt)
    se = metric.binary.sensitivity(pred, gt)
    sp = metric.binary.specificity(pred, gt)
    label_gt = label(gt)
    label_gts = np.bincount(label_gt.flat)
    label_pred = label(pred)
    label_preds = np.bincount(label_pred.flat)
    M, N = label_gts.shape[0], label_preds.shape[0]
    index = np.where(label_gts<11)
    idx_offset = 0
    if index[0].size !=0:
        for idx in range(index[0].shape[0]):
            mask = label_gt==index[0][idx]-idx_offset
            label_gt[mask]=0
            # we need to close the gap after removing the label
            label_gt[label_gt>index[0][idx]-idx_offset] -=1
            idx_offset += 1
            M=M-1
    index = np.where(label_preds<11)
    idx_offset = 0
    if index[0].size !=0:
        for idx in range(index[0].shape[0]):
            mask = label_pred==index[0][idx]-idx_offset
            label_pred[mask]=0
            # we need to close the gap after removing the label
            label_pred[label_pred>index[0][idx]-idx_offset] -=1
            idx_offset += 1
            N=N-1
    H_ij = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            H_ij[i, j] = ((label_gt==i) * (label_pred==j)).sum()
    TPg=0
    for i in range(1, M):
        alpha = H_ij[i, 1:].sum() / (H_ij[i, :].sum() + 1e-18)
        if alpha > 0.1:
            wsum, k, vaccept=0, 0, True
            while wsum < 0.65:
                pk = np.argsort(-H_ij[i, 1:])[k]+1#np.argwhere(np.argsort(H_ij[i])==k)[0][0]
                tk = H_ij[0, pk] / H_ij[:, pk].sum()
                if tk >0.7:
                    vaccept = False
                    break
                wsum += H_ij[i, pk] / H_ij[i, 1:].sum()
                k +=1
            if vaccept == True:
                TPg +=1
    TPa=0
    H_ji = H_ij.T
    for j in range(1, N):
        alpha = H_ji[j, 1:].sum() / (H_ji[j, :].sum()+ 1e-18)
        if alpha > 0.1:
            wsum, k, vaccept=0, 0, True
            while wsum < 0.65:
                pk = np.argsort(-H_ji[j, 1:])[k]+1#np.argwhere(np.argsort(H_ji[j])==k)[0][0]
                tk = H_ji[0, pk] / H_ji[:, pk].sum()
                if tk >0.7:
                    vaccept = False
                    break
                wsum += H_ji[j, pk] / H_ji[j, 1:].sum()
                k +=1
            if vaccept == True:
                TPa +=1
    sel, pl = TPg/(M-1),TPa/(N-1)
    if sel == 0 or pl == 0:
        F1 = 0
    else:
        F1 = (2 * sel * pl) / (sel+pl)
    print("TPg:{}, M:{}, TPa:{}, N:{}".format(TPg, M-1, TPa, N-1))
    print("sel:{}, pl:{}, f1:{}".format(sel, pl, F1))
    print("dice:{}, jc:{}, 95hd:{}, asd:{}, pr:{}, se:{}, sp:{}".format(dice, jc, hd, asd, precision, se, sp))
    return dice, jc, hd, asd, precision, se, sp, F1