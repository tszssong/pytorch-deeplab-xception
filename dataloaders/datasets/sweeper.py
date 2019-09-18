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
DataPath = collections.namedtuple('DataPath', ['imgpath', 'deppath', 'labpath'])
ImageFile.LOAD_TRUNCATED_IMAGES = True

class sweeperSegmentation(Dataset):
    NUM_CLASSES = 2
    CAT_LIST = [0, 62]

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('sweeper'),
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
            if 'EH' in imgname:
                deppath = os.path.join(base_dir, 'images/{}{}'.format(split, year), imgname.replace('EH.png', 'DP.png'))
                labpath = os.path.join(base_dir, 'images/{}{}'.format(split, year), imgname.replace('EH.png', 'EH1.png'))
            else:
                deppath = os.path.join(base_dir, 'images/{}{}'.format(split, year), imgname.replace('IR.png', 'De.png'))
                labpath = os.path.join(base_dir, 'images/{}{}'.format(split, year), imgname.replace('IR.png', 'label.png'))
                # print(imgname, deppath, labpath)
            self.ids.append(DataPath(imgpath=imgpath, deppath=deppath, labpath=labpath))
        
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
        _img = Image.open(imgpath).convert('RGB')
        r1,g1,b1 = _img.split()        
        deppath = self.ids[index].deppath
        _depth = Image.open(deppath).convert('RGB')
        r2, g2, b2 = _depth.split()
        tmp= [r1, b1, r2]
        img = Image.merge("RGB",tmp)
        
        tgtpath = self.ids[index].labpath
        mask = Image.open(tgtpath).convert('L')
        label = np.array(mask)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] ==255 or label[i][j]==254:
                    label[i][j] = 0
                else:
                    label[i][j] = 1
        _target = Image.fromarray(label)
        # plt.subplot(231)
        # plt.imshow(_img)
        # plt.subplot(232)
        # plt.imshow(r1)
        # plt.subplot(234)
        # plt.imshow(_depth)
        # plt.subplot(235)
        # plt.imshow(r2)
        # plt.subplot(236)
        # plt.imshow(img)
        # plt.subplot(233)
        # plt.imshow(_target)
        # plt.show()
        return img, _target


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixScaleCrop(crop_size=self.args.crop_size),#tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
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
    # args.base_size = 513
    # args.crop_size = 513
    args.base_size = 257
    args.crop_size = 257

    sweeper_val = sweeperSegmentation(args, split='train', year='2019')
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

        # if ii == 10:
        #     break

    plt.show(block=True)
