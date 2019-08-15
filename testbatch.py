import argparse
import os, time
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
def test(args):
    
    val_set = sweeper.sweeperSegmentation(args, split='val')
    val_set = sweeper.sweeperSegmentation(args, split='train')
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, **kwargs)
        
    model = DeepLab(num_classes=2,
                        backbone='resnet',
                        output_stride=16)
    model = model.cuda()
    # checkpoint = torch.load('run/sweeper_bgr/deeplab-resnet//model_best.pth.tar')
    checkpoint = torch.load('run/sweeper/deeplab-resnet//model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
    test_loss = 0.0
    start = time.time()
    for i, sample in enumerate(val_loader):
        image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        print("use time:%.5f"%(time.time()-start))
        loss = criterion(output, target)
        test_loss += loss.item()
        print('Test loss: %.5f, average loss: %.5f' % (loss.item(), test_loss / (i + 1)))
        sys.stdout.flush()
        # pred = output.data.cpu().numpy()
        # target = target.cpu().numpy()
        # pred = np.argmax(pred, axis=1)
        
        # input = image.cpu().numpy()[0].transpose((1,2,0))
        # input *= (0.229, 0.224, 0.225)
        # input += (0.485, 0.456, 0.406)
        # input *= 255.0
        # input = input.astype(np.uint8)
        # # cv2.imshow("input",input)
        # # cv2.imshow("label", gt)
        # # cv2.imshow("output", pred)
        # gt = target*255
        # gt = gt.transpose((1,2,0)).astype(np.uint8)
        # pred = pred*255
        # pred = pred.transpose((1,2,0)).astype(np.uint8)
        # show_gt = cv2.resize(gt, (200,200))
        # show_pr = cv2.resize(pred, (200,200))
        # show_in = cv2.resize(input, (200,200))
        # # print(show_in.shape, show_gt.shape, show_pr.shape) 
        # htitch = np.hstack((show_gt, show_in[:,:,0], show_pr)) #input has 3 same channels 
        # cv2.imshow("h", htitch)
        # cv2.waitKey(1)
        # cv2.imwrite('/home/ubuntu/zms/data/sweeper/output/%04d.bmp'%i,htitch)

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
