# --------------------------------------------------------
# ASM
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xiaopeng Yan
# --------------------------------------------------------

from __future__ import division
import _init_paths
import os
import logging
import cv2
import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect,im_predict
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import random
CLASSES=('__background__','aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
def softmax(ary):
    ary = ary.flatten()
    expa = np.exp(ary)
    dom = np.sum(expa)
    return expa/dom

def choose_model(dir):
    '''
    get the latest model in in dir'''
    lists = os.listdir(dir)
    lists.sort(key=lambda fn:os.path.getmtime(os.path.join(dir,fn)))
    return lists

def load_model(net_file ,path):
    '''
    return caffe.Net'''
    import caffe
    net = caffe.Net(net_file, path, caffe.TEST)    
    return net
def judge_y(score):
    '''return :
    y:np.array len(score)
    '''
    y=[]
    for s in score:
        if s==1 or np.log(s)>np.log(1-s):
            y.append(1)
        else:
            y.append(-1)
    return np.array(y, dtype=np.int)


def bulk_detect(net, detect_idx, imdb, clslambda):
    '''
    return 
    scoreMatrix: len(detect_idx) * R * K
    boxRecord: len(detext_idx) * R * K * 4
    eps: max(1-l/lambda)
    '''
    import cv2
    from fast_rcnn.config import cfg
    from utils.timer import Timer
    from fast_rcnn.nms_wrapper import nms
    from fast_rcnn.test import im_detect

    roidb = imdb.roidb
    allBox =[]; allScore = []; eps = 0; allY=[]; al_idx = []
    for i in detect_idx:
        imgpath = imdb.image_path_at(i)
        im = cv2.imread(imgpath)
        height = im.shape[0]; width=im.shape[1]

        timer = Timer()
        timer.tic()
        scores, boxes = im_predict(net, im)
        timer.toc()

        BBox=[] # all eligible boxes for this img
        Score=[] # every box in BBox has k*1 score vector
        Y = [] # every box in BBox has k*1 cls vector
        CONF_THRESH = 0.5 
        NMS_THRESH = 0.3
        if np.amax(scores[:,1:])<CONF_THRESH:
            al_idx.append(i)
            continue
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4:8]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                        cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

            if len(inds) == 0 :
                continue
        #    vis_detections(imgpath.split('/')[-1], im, cls, dets, thresh=CONF_THRESH)
            for j in inds:
                bbox = dets[j, :4]
                BBox.append(bbox)
                # find which region this box deriving from
                k = keep[j]
                Score.append(scores[k].copy())
                Y.append(judge_y(scores[k]))
                y = Y[-1]
                loss = -( (1+y)/2 * np.log(scores[k]) + (1-y)/2 * np.log(1-scores[k]+(1e-30)))
                tmp = np.max(1-loss/clslambda)
                eps = eps if eps >= tmp else tmp

        allBox.append(BBox[:]); allScore.append(Score[:]); allY.append(Y[:])
    return np.array(allScore), np.array(allBox), np.array(allY), al_idx, eps


def judge_uv(loss, gamma, clslambda, eps):
    '''
    return 
    u: scalar
    v: R^kind vector
    '''
    lsum = np.sum(loss)
    dim = loss.shape[0]
    v = np.zeros((dim,))
    
    if(lsum>gamma):
        return 1, np.array([eps]*dim)
    elif lsum<gamma:
        for i,l in enumerate(loss):
            if l>clslambda[i]:
                v[i]=0
            elif l<clslambda[i]*(1-eps):
                v[i]=eps
            else:
                v[i]=1-l/clslambda[i]
    return 0, v

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    plt.switch_backend('Agg')
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig,ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # save the image
    fig = plt.gcf()
    import time
    t0 = time.time()
    fig.savefig(str(t0)+'.jpg')


