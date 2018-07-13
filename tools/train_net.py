#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
from __future__ import division
import _init_paths
from fast_rcnn.train import get_training_roidb, train_net, SolverWrapper, update_training_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
from utils.help import *
import caffe
import argparse
import pprint
import numpy as np
import sys, math, logging
import scipy
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    ############## begin #########################
    parser.add_argument('--enable_al', help='do not use al process',
                        action='store_true',default=True)
    parser.add_argument('--enable_ss', help='do not use ss process',
                        action='store_true',default=True)
    ############## end ############################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print ('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print ('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

################# begin #########################
def get_Imdbs(imdb_names):
    imdbs = [get_imdb(s) for s in imdb_names.split('+')]
    for im in imdbs:
        im.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print ('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    return datasets.imdb.Imdbs(imdbs)

from bitmap import BitMap
################# end ###########################
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
############################ begin ##########################################
    imdb = get_Imdbs(args.imdb_name)
    roidb = get_training_roidb(imdb)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # some statistic to record
    altotal = 0; sstotal = 0
    ignoretotal = 0
    # set bitmap for AL
    bitmapImdb = BitMap(imdb.num_images)
    # choose initiail samples:VOC2007
    initial_num = len(imdb[imdb.item_name(0)].roidb)
    print 'All VOC2007 images use for initial train, image numbers:%d'%(initial_num)
    for i in range(initial_num):
        bitmapImdb.set(i)

    train_roidb = [roidb[i] for i in range(initial_num)]
    pretrained_model_name = args.pretrained_model

    # static parameters
    tao = args.max_iters
    # initial hypeparameters
    gamma = 0.15; clslambda = np.array([-np.log(0.9)]*imdb.num_classes)
    # train record
    epochcounter = 0; train_iters = 0; iters_sum = train_iters
    # control al proportion
    al_proportion_checkpoint = [int(x*initial_num) for x in np.linspace(0.3,2.0,10)]
    # control ss proportion with respect to al proportion
    ss_proportion_checkpoint = [int(x*initial_num) for x in np.linspace(0.2,4,10)]
    
    
    # get solver object
    sw = SolverWrapper(args.solver, train_roidb, output_dir,
                        pretrained_model=pretrained_model_name)
    # with voc2007 to pretrained an initial model
#    sw.train_model(70000)

    while(True):
        # detact unlabeledidx samples
        unlabeledidx = list(set(range(imdb.num_images))-set(bitmapImdb.nonzero()))
        # detect labeledidx
        labeledidx = list(set(bitmapImdb.nonzero()))
        # load latest trained model
        trained_models = choose_model(output_dir)
        pretrained_model_name = trained_models[-1] 
        modelpath = os.path.join(output_dir, pretrained_model_name)
        protopath = os.path.join('models/pascal_voc/ResNet-101/rfcn_end2end',
                'test_agnostic.prototxt')
        print 'choose latest model:{}'.format(modelpath)
        model = load_model(protopath,modelpath)
        print('Process detect the unlabeled images ...')
        # return detect results of the unlabeledidx samples with the latest model
        scoreMatrix, boxRecord,yVecs, al_idx, eps = bulk_detect(model, unlabeledidx, imdb, clslambda)
     #   logging.debug('scoreMatrix:{}, boxRecord:{}, eps:{}, yVecs:{}'.format(scoreMatrix.shape,
     #       boxRecord.shape, eps, yVecs.shape))
        unlabeledidx = [ x for x in unlabeledidx if x not in al_idx ]
        # record some detect results for updatable
        al_candidate_idx = [] # record al samples index in imdb
        ss_candidate_idx = [] # record ss samples index in imdb
        ss_fake_gt = [] # record fake labels for ss
        cls_loss_sum = np.zeros((imdb.num_classes,)) # record loss for each cls
        count_box_num = 0 # used for update clslambda
        print('Process Self-supervised Sample Mining ...')
        for i in tqdm(range(len(unlabeledidx))):
            img_boxes = []; cls=[]; # fake ground truth
            count_box_num += len(boxRecord[i])
            for j,box in enumerate(boxRecord[i]):
                boxscore = scoreMatrix[i][j] # score of a box
                # fake label box
                y = yVecs[i][j]
                # the fai function
                loss = -((1+y)/2 * np.log(boxscore) + (1-y)/2 * np.log(1-boxscore+1e-30))
                cls_loss_sum += loss
                # choose u,v by loss
                u, v = judge_uv(loss, gamma, clslambda, eps)
                # SS process 
                if(u!=1):
                    if i % 10000 == 0:
                        print('SS Processing ... ')
                    if(np.sum(y==1)==1 and np.where(y==1)[0]!=0): # not background
                            # add fake gt
                            img_boxes.append(box)
                            cls.append(np.where(y==1)[0])
                    elif(np.sum(y==1)!=1):
                         ignoretotal += 1
                else: # AL process
                    if i%10000 == 0:
                        print('AL Processing ... ')
                    #add image to al candidate
                    al_candidate_idx.append(unlabeledidx[i])
                    img_boxes=[]; cls=[]
                    break
            # replace the fake ground truth for the ss_candidate
            if len(img_boxes) != 0:
                ss_candidate_idx.append(unlabeledidx[i])
                overlaps = np.zeros((len(img_boxes), imdb.num_classes), dtype=np.float32)
                for i in range(len(img_boxes)):
                    overlaps[i, cls[i]]=1.0

                overlaps = scipy.sparse.csr_matrix(overlaps)
                ss_fake_gt.append({'boxes':np.array(img_boxes),
                    'gt_classes':np.array(cls,dtype=np.int).flatten(),
                    'gt_overlaps':overlaps, 'flipped':False})

        if len(al_candidate_idx)<=10 or iters_sum>args.max_iters:
            print ('all process finish at loop ',epochcounter)
            print ('the num of al_candidate :',len(al_candidate_idx))
            print ('the net train for {} epoches'.format(iters_sum))
            break
        # 50% enter al
        r = np.random.rand(len(al_candidate_idx))
        al_candidate_idx = [x for i,x in enumerate(al_candidate_idx) if r[i]>0.5]

        if args.enable_al:
            # control al proportion
            print('altotal:',altotal,'al_candidate_idx:',len(al_candidate_idx),'al_proportion_checkpoint:',al_proportion_checkpoint[0])
            if altotal+len(al_candidate_idx)>=al_proportion_checkpoint[0]:
                al_candidate_idx = al_candidate_idx[:int(al_proportion_checkpoint[0]-altotal)]
                tmp = al_proportion_checkpoint.pop(0)
                al_proportion_checkpoint.append(tmp)
                print 'al_proportion_checkpoint: {}%% samples for al, model name:{}'.format(tmp/initial_num,pretrained_model_name )
            
            print 'sample chosen for al: ', len(al_candidate_idx)
        else:
            al_candidate_idx = []
        if args.enable_ss:
            # control ss proportion
            print('sstotal:',sstotal,'ss_candidate_idx:',len(ss_candidate_idx),'ss_proportion_checkpoint:',ss_proportion_checkpoint[0])
            if sstotal+len(ss_candidate_idx)>=ss_proportion_checkpoint[0]:
                ss_candidate_idx = ss_candidate_idx[:int(ss_proportion_checkpoint[0]-sstotal)]
                ss_fake_gt = ss_fake_gt[:int(ss_proportion_checkpoint[0]-sstotal)]
                tmp = ss_proportion_checkpoint.pop(0)
                ss_proportion_checkpoint.append(tmp)
                print 'ss_proportion_checkpoint: {}%% samples for ss, model name:{}'.format(tmp/initial_num,pretrained_model_name )
            
            print 'sample chosen by ss: ',len(ss_candidate_idx)
        else:
            ss_candidate_idx=[]
            ss_fake_gt = []
        print 'sample ignore:', ignoretotal
        altotal += len(al_candidate_idx); sstotal += len(ss_candidate_idx)+ignoretotal
        # record the proportion of al and ss
        al_factor = float(altotal/initial_num)
        ss_factor = float(sstotal/initial_num)
        logging.info('last model name :{},al amount:{}/{},al_factor:{},ss amount: {}/{},ss_factor:{}'.format(pretrained_model_name,altotal,initial_num,al_factor,sstotal,initial_num,ss_factor))
        # generate training set for next loop
        for idx in al_candidate_idx:
            bitmapImdb.set(idx)
        next_train_idx = bitmapImdb.nonzero(); next_train_idx.extend(ss_candidate_idx)

        # cfg.TRAIN.USE_FLIPPED = False # dont need filp again
        # update the roidb with ss_fake_gt 
        roidb = update_training_roidb(imdb,ss_candidate_idx,ss_fake_gt)
        train_roidb = [roidb[i] for i in next_train_idx]
 
        # stop condition
        epochcounter += 1
        if iters_sum<=tao:
           clslambda = 0.9 * clslambda - 0.1*np.log(softmax(cls_loss_sum/(count_box_num+1e-30)))
           gamma = min(gamma+0.05,1)
           cls_loss_sum = 0.0       

        # add the labeled samples to finetune W
        train_iters = 20000
        iters_sum += train_iters
        sw.update_roidb(train_roidb)
        sw.train_model(iters_sum)

############################## end #########################################




