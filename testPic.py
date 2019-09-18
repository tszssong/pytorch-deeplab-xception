import argparse
import os, time, shutil
import cv2
import numpy as np
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from PIL import Image
from dataloaders.datasets import cityscapes, coco, sweeper, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile


def test(args):
    size = args.crop_size
    
    # composed_transforms = transforms.Compose([
    #     tr.FixScaleCrop(crop_size=args.crop_size),
    #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     tr.ToTensor()])
    composed_transforms = transforms.Compose([ transforms.Resize([size,size]), transforms.ToTensor(), 
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    val_set = sweeper.sweeperSegmentation(args, split='val')
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        
    model = DeepLab(num_classes=2, backbone='resnet', output_stride=16)
    model = model.cuda()
    checkpoint = torch.load('run/sweeper/deeplab-resnet//model_best.pth.tar')
    # checkpoint = torch.load('run/sweeper/deeplab-resnet/experiment_4/checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
    test_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # with open('/home/ubuntu/zms/data/sweeper/depth_320x200/black_tile_01.txt') as fp:
    # with open('/home/ubuntu/zms/data/sweeper/depth_320x200/write_tile_04.txt') as fp:
    with open('/home/ubuntu/zms/data/sweeper/sweeper_320x200.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            start = time.time()
            imgpath = line.strip()
            deppath = imgpath.replace('EH.png', 'De.png')
            # image = Image.open(imgpath)

            _img = Image.open(imgpath).convert('RGB')
            _depth = Image.open(deppath).convert('RGB')

            r1,g1,b1 = _img.split()
            r2, g2, b2 = _depth.split()
            tmp= [r1, b1, r2]
            image = Image.merge("RGB",tmp)

            print(image.size)
            im_width = image.size[0]
            im_hight = image.size[1]
            dir_name = imgpath.split('/')[-2]
            img_name = imgpath.split('/')[-1]
            image = image.convert('RGB')
            image = composed_transforms(image)
          
            im = image.reshape(1,3,size,size)
            
            with torch.no_grad():
                output = model(im.to(device))
       
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = pred*255
            pred = pred.transpose((1,2,0)).astype(np.uint8)
              
            print("time used:%.2f"%(time.time()-start))
            input = image.cpu().numpy().transpose((1,2,0))
            input *= (0.229, 0.224, 0.225)
            input += (0.485, 0.456, 0.406)
            input *= 255.0
            input = input.astype(np.uint8)

            # cv2.imshow("input",input)
            # cv2.imshow("pred", pred)
            # cv2.waitKey()
            # savepath = '/home/ubuntu/zms/data/sweeper/depth_320x200/out/' + dir_name
            savepath = '/home/ubuntu/zms/data/sweeper/320x200out/' + dir_name
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            output = cv2.resize(pred, (im_width, im_hight))
            # shutil.copy2(imgpath, savepath +'/'+ img_name)
            img_name= img_name.replace('.png','.bmp')
            cv2.imwrite(savepath+'/'+img_name,output)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
   
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    test(args)
if __name__ == "__main__":
   main()
