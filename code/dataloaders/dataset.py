import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import numpy as np

def random_rot_flip(image_1, image_2, label):
    k = np.random.randint(0, 4)
    image_1 = np.rot90(image_1, k)
    image_2 = np.rot90(image_2, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image_1 = np.flip(image_1, axis=axis).copy()
    image_2 = np.flip(image_2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image_1, image_2, label

def random_rotate(image_1, image_2, label):
    angle = np.random.randint(-20, 20)
    image_1 = ndimage.rotate(image_1, angle, order=0, reshape=False)
    image_2 = ndimage.rotate(image_2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image_1, image_2, label

class MS(Dataset):
    """ MS Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        if "ms23" in image_name:
            h5f = h5py.File(image_name, 'r')
            image_1 = h5f['image'][:]
            image_2 = h5f['image'][:]
            label = h5f['label'][:]
        else:
            h5f = h5py.File(image_name, 'r')
            image_1 = h5f['image_1'][:]
            image_2 = h5f['image_2'][:]
            label = h5f['label'][:]
        sample = {'image_1': image_1, 'image_2': image_2, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class WeightCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_1, image_2, label = sample['image_1'], sample['image_2'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image_1 = np.pad(image_1, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image_2 = np.pad(image_2, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image_1.shape
        if label.sum() > 0:
            mask = np.nonzero(label)
            num_label_pixel = mask[0].shape[0]
            center_index =np.random.randint(0, num_label_pixel-1)
            center_x, center_y, center_z = mask[0][center_index], mask[1][center_index], mask[2][center_index]
            w1 = np.random.randint(-10, 10)+self.output_size[0]//2
            h1 = np.random.randint(-10, 10)+self.output_size[1]//2
            d1 = np.random.randint(-10, 10)+self.output_size[2]//2
            lefttop_x, lefttop_y, lefttop_z = center_x-w1, center_y-h1, center_z-d1
            minx = max(lefttop_x, 0)
            miny = max(lefttop_y, 0)
            minz = max(lefttop_z, 0)
            maxx = minx +  self.output_size[0]
            maxy = miny +  self.output_size[1]
            maxz = minz +  self.output_size[2]
            if maxx>= w or maxy >= h or maxz >=d:
                maxx = min(maxx, w-1)
                maxy = min(maxy, h-1)
                maxz = min(maxz, d-1)
                minx = maxx - self.output_size[0]
                miny = maxy - self.output_size[1]
                minz = maxz - self.output_size[2]
            label = label[minx:maxx, miny:maxy, minz:maxz]
            image_1 = image_1[minx:maxx, miny:maxy, minz:maxz]
            image_2 = image_2[minx:maxx, miny:maxy, minz:maxz]
            assert(label.shape == self.output_size)
        else:
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image_1 = image_1[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image_2 = image_2[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            assert(label.shape == self.output_size)
        return {'image_1': image_1, 'image_2': image_2, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image_1, image_2, label = sample['image_1'], sample['image_2'], sample['label']
        image_1, image_2, label = random_rot_flip(image_1, image_2, label)

        return {'image_1': image_1, 'image_2': image_2, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image_1, image_2, label = sample['image_1'], sample['image_2'], sample['label']
        image_1, image_2, label = random_rotate(image_1, image_2, label)

        return {'image_1': image_1, 'image_2': image_2, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_1, image_2, label = sample['image_1'], sample['image_2'], sample['label']
        image_1 = image_1.reshape(1, image_1.shape[0], image_1.shape[1], image_1.shape[2]).astype(np.float32)
        image_2 = image_2.reshape(1, image_2.shape[0], image_2.shape[1], image_2.shape[2]).astype(np.float32)

        return {'image_1': torch.from_numpy(image_1), 'image_2': torch.from_numpy(image_2), 'label': torch.from_numpy(label).long()}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, primary_batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = primary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
