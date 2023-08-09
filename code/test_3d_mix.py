import os
import argparse
import torch
from networks.net_factory import net_factory
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,  default='CoactSeg', help='name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='reg', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = FLAGS.root_path + "model/{}_{}/{}".format(FLAGS.name, FLAGS.exp, FLAGS.model)
test_save_path = FLAGS.root_path + "model/{}_{}/{}_predictions/".format(FLAGS.name, FLAGS.exp, FLAGS.model)

num_classes = 2

patch_size = (80, 80, 80)
FLAGS.root_path = FLAGS.root_path + 'data/'
with open(FLAGS.root_path + '/val.list', 'r') as f:
    image_list = f.readlines()
image_list = [item.replace('\n','') for item in image_list]
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    
    net = net_factory(net_type=FLAGS.model, in_chns=3, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                    patch_size=(80, 80, 80), stride_xy=20, stride_z=20,
                    save_result=True, test_save_path=test_save_path,
                    metric_detail=FLAGS.detail, nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
