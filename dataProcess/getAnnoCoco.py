import sys, os
import numpy as np
import cv2
wkdir = '/home/ubuntu/zms/data/seg/'
f = open(wkdir+'/train2019.txt','r')
for line in f.readlines():
    imgname = line.strip().split('\n')[0]
    oriimgpath = wkdir + 'train2019/'+imgname
    segimgpath = wkdir + 'train2019/' + imgname.replace("EH.pn","EH1.pn")
    ori = cv2.imread(oriimgpath)
    seg = cv2.imread(segimgpath)
    cv2.imshow("ori", ori)
    cv2.imshow("seg", seg)
    cv2.waitKey()    
    if not np.array_equal(ori.shape, seg.shape):
        print (ori.shape, seg.shape)
    #if not 
    #print (ori[10][10][0], ori[10][10][1], ori[10][10][2], seg[10][10][0], seg[10][10][1],seg[10][10][2])
