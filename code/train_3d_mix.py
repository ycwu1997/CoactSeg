import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import ramps, losses, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,  default='CoactSeg', help='name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='reg', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=20000, help='maximum iteration to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--patch_size', type=int,  default=80, help='the size of patch')

args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}/{}".format(args.name, args.exp, args.model)

num_classes = 2
patch_size = (args.patch_size, args.patch_size, args.patch_size)
args.root_path = args.root_path+'data/'
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
max_iterations = args.max_iteration
base_lr = args.base_lr
batch_size = args.batch_size

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes, mode="train")
    db_train = MS(base_dir=train_data_path,
                    split='train',
                    transform = transforms.Compose([
                        RandomRotFlip(),
                        RandomRot(),
                        WeightCrop(patch_size),
                        ToTensor(),
                        ]))
    labeled_idxs = list(range(32))
    unlabeled_idxs = list(range(32, 62))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, pin_memory=True, num_workers=2)
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch_1, volume_batch_2, label_batch = sampled_batch['image_1'], sampled_batch['image_2'], sampled_batch['label']
            volume_batch_sub = volume_batch_2 - volume_batch_1
            volume_batch = torch.cat([volume_batch_1, volume_batch_2, volume_batch_sub], dim=1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs_1, outputs_2, outputs_3 = model(volume_batch)

            y1 = F.softmax(outputs_1, dim=1)
            y2 = F.softmax(outputs_2, dim=1)
            y3 = F.softmax(outputs_3, dim=1)

            # for public dataset
            # only containing the new lesion labels
            label_new_lesions = label_batch[:batch_size,...]

            loss_seg_public = F.cross_entropy(outputs_3[:batch_size,...], label_new_lesions)
            loss_seg_dice_public = losses.Binary_dice_loss(y3[:batch_size,1,...], label_new_lesions == 1)

            selected_new_lesions_y1 = torch.masked_select(y1[:batch_size,1,...], label_new_lesions==1)
            selected_new_lesions_y2 = torch.masked_select(y2[:batch_size,1,...], label_new_lesions==1)

            selected_new_lesions_gt = torch.masked_select(label_new_lesions, label_new_lesions==1)

            loss_reg_pseudo = losses.mse_loss(selected_new_lesions_y1, 1-selected_new_lesions_gt) + losses.mse_loss(selected_new_lesions_y2, selected_new_lesions_gt)

            # for inhouse dataset
            # containing all lesions
            loss_seg_1 = F.cross_entropy(outputs_1[batch_size:,...], label_batch[batch_size:,...])
            loss_seg_2 = F.cross_entropy(outputs_2[batch_size:,...], label_batch[batch_size:,...])

            loss_seg_dice_1 = losses.Binary_dice_loss(y1[batch_size:,1,...], label_batch[batch_size:,...] == 1)
            loss_seg_dice_2 = losses.Binary_dice_loss(y2[batch_size:,1,...], label_batch[batch_size:,...] == 1)

            loss_seg_inhouse = loss_seg_1 + loss_seg_2
            loss_seg_dice_inhouse = loss_seg_dice_1 + loss_seg_dice_2

            loss_reg_inhouse = losses.mse_loss(y1[batch_size:,...], y2[batch_size:,...])
            
            iter_num = iter_num + 1
            
            loss_seg = loss_seg_inhouse + loss_seg_public
            loss_reg = loss_reg_inhouse + loss_reg_pseudo
            loss_seg_dice = loss_seg_dice_inhouse +  loss_seg_dice_public

            writer.add_scalar('loss/1_loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/2_loss_seg_ce', loss_seg, iter_num)
            writer.add_scalar('loss/3_loss_seg_reg', loss_reg, iter_num)
            
            if iter_num < 10000:
                loss = loss_seg_dice
            else:
                loss = loss_seg_dice + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_seg_dice: %03f, loss_seg_ce: %03f, loss_reg: %03f' % (iter_num, loss, loss_seg_dice, loss_seg, loss_reg))

            if iter_num >= 0:#200 and iter_num % 200 == 0:
                sample_index = np.random.randint(0, batch_size)
                img_double = ramps.get_imgs(y1, y2, y3, volume_batch, label_batch, sample_index)
                writer.add_images('Epoch_%d_Iter_%d_Double'% (epoch_num, iter_num), img_double)
                sample_index = np.random.randint(batch_size, 2*batch_size)
                img_single = ramps.get_imgs(y1, y2, y3, volume_batch, label_batch, sample_index)
                writer.add_images('Epoch_%d_Iter_%d_Single'% (epoch_num, iter_num), img_single)

            if iter_num >= 0:#5000 and iter_num % 200 == 0:
                model.eval()
                dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=20, stride_z=20)
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()