import argparse
import os
import cv2
import numpy as np

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.datasets import cityscapes, coco, sweeper, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader
def test(args):
    
    val_set = sweeper.sweeperSegmentation(args, split='val')
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        
    model = DeepLab(num_classes=2,
                        backbone='resnet',
                        output_stride=16)
    model = model.cuda()
    checkpoint = torch.load('run/sweeper/deeplab-resnet//model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
    test_loss = 0.0
    for i, sample in enumerate(val_loader):
        image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, target)
        test_loss += loss.item()
        print('Test loss: %.3f, average loss: %.3f' % (loss.item(), test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        print(pred.shape, pred[0].shape, image.shape, target.shape)
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        show_re = pred[0]
        print(show_re.shape)
        show_re = show_re-np.amin(show_re)
        show_re = show_re/np.amax(show_re)
        show_re = show_re*255
        show_re = show_re.astype(np.uint8)
        gt = target
        gt = gt-np.amin(gt)
        gt = gt/np.amax(gt)
        gt = gt*255
        gt = gt.transpose((1,2,0)).astype(np.uint8)
        input = image.cpu().numpy()[0].transpose((1,2,0))
        input *= (0.229, 0.224, 0.225)
        input += (0.485, 0.456, 0.406)
        input *= 255.0
        input = input.astype(np.uint8)
        cv2.imshow("input",input)
        cv2.imshow("label", gt)
        cv2.imshow("output", show_re)
        show_re = np.expand_dims(show_re, axis=0)
        show_re = show_re.transpose((1,2,0)).astype(np.uint8)
        show_gt = cv2.resize(gt, (200,200))
        show_re = cv2.resize(show_re, (200,200))
        show_in = cv2.resize(input, (200,200))
        print(show_in[0].shape, show_gt.shape, show_re.shape)  #input has 3 same channels
        htitch = np.hstack((show_gt, show_in[:,:,0], show_re))
        cv2.imshow("h", htitch)
        cv2.waitKey()

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
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
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
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
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

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    test(args)
if __name__ == "__main__":
   main()
