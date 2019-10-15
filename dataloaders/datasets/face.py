import os
import sys
sys.path.append('../../')
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
import collections
DataPath = collections.namedtuple('DataPath', ['imgpath', 'labpath'])
ImageFile.LOAD_TRUNCATED_IMAGES = True

class faceSegmentation(Dataset):
    NUM_CLASSES = 2
    CAT_LIST = [0, 62]

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('face'),
                 split='train',
                 year='2019'):
        super().__init__()
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.ids = []

        fread = open(base_dir + split + year + '.txt')
        for line in fread.readlines():
            imgname = line.strip()
            imgpath = os.path.join(base_dir, 'images/{}{}'.format(split, year), imgname)
            labname = imgname.replace('.jpg', '.png')
            labpath = os.path.join(base_dir, 'images/{}{}'.format(split, year), labname)
            self.ids.append(DataPath(imgpath=imgpath, labpath=labpath))
        
        self.args = args

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        imgpath = self.ids[index].imgpath
        img = Image.open(imgpath).convert('RGB')
       
        tgtpath = self.ids[index].labpath
        mask = Image.open(tgtpath).convert('L')
        label = np.array(mask)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                #if label[i][j] == 0 or label[i][j]==18:  #cloth
                if label[i][j] == 0 or label[i][j]==16 or label[i][j]==17 or label[i][j]==18:  #neck+cloth
                    label[i][j] = 0
                else:
                    label[i][j] = 1
        _target = Image.fromarray(label)
        return img, _target


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    sweeper_val = faceSegmentation(args, split='train', year='2019')
    dataloader = DataLoader(sweeper_val, batch_size=1, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 10:
            break

    plt.show(block=True)
