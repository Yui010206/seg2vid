from  __future__ import division
import math
import os
import numpy as np
from PIL import Image
import imageio
import cv2
import torch
import sys
import matplotlib.pyplot as plt
from cv2 import Stitcher
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import json
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

SHT_LABEL = '/mnt/lustreold/yushoubin/data/VAD/ShanghaiTech/test_frame_mask/'
#SHT_LABEL = '/Users/yushoubin/Desktop/VAD_DATA/ShanghaiTech/test_frame_mask/'


def save_json(data,iteration,name,save_dir):
    save_path = save_dir + '/' + str(iteration)+ '_' + name + '.json'
    with open(save_path,'w') as f:
        f.write(json.dumps(data))

def compute_rec_error(rec,gt):
    # rec, gt: [B,N,2]

    err = torch.abs(rec-gt)
    b,n,c = err.shape
    err = err.reshape(b,-1)
    err = torch.sum(err,dim=1)

    return err.tolist()

def recover_pose(local_pose,global_box,opt):
    #print(local_pose.shape)
    #print(global_box.shape)
    xmin,ymin,xmax,ymax = [:,:,0],[:,:,1],[:,:,2],[:,:,3]
    w = xmax - xmin
    h = ymax - ymin
    center_x, center_y = (xmax + xmin)/2, (ymax + ymin)/2
    mean_w = torch.mean(w,dim=-1)
    mean_h = torch.mean(h,dim=-1)

    if opt.mean_box:
        local_pose_x = local_pose[:,:,0]
        local_pose_y = local_pose[:,:,1]
 
    else:
        
        w, h = global_box[:,1,0], global_box[:,1,1]
        xmin, ymin = center_x - w/2, center_y - h/2

    w = w.unsqueeze(1).repeat(1,local_pose_x.shape[1])
    h = h.unsqueeze(1).repeat(1,local_pose_x.shape[1])
    xmin = xmin.unsqueeze(1).repeat(1,local_pose_x.shape[1])
    ymin = ymin.unsqueeze(1).repeat(1,local_pose_x.shape[1])

    global_pose_x = local_pose_x*w + center_x
    global_pose_y = local_pose_y*h + center_y

    #print(global_pose_y.shape)
    global_pose = torch.cat([global_pose_x.unsqueeze(-1),global_pose_y.unsqueeze(-1)],-1)
    #print(global_pose.shape)

    return global_pose

def compute_rec_error(pred,gt,opt,gt_track=True):
    # rec, gt: [B,N,2]

    if opt.headless:
        pred_pose = pred[:,-14:,:]
        gt_pose = gt[:,-14:,:]
        gt_box = gt[:,-16:-14,:]

        if gt_track:
            pred_box = gt[:,-16:-14,:]
        else:
            pred_box = pred[:,-16:-14,:]
        
    else:

        pred_pose = pred[:,-17:,:]
        gt_pose = gt[:,-17:,:]
        gt_box = gt[:,-19:-17,:]

        if gt_track:
            pred_box = gt[:,-19:-17,:]
        else:
            pred_box = pred[:,-19:-17,:]

    gt_pose_recover = recover_pose(gt_pose,gt_box,opt)
    predict_pose_recover = recover_pose(pred_pose,pred_box,opt)

    err = torch.abs(gt_pose_recover-predict_pose_recover)
    # err (b,n,2)
    b,n,c = err.shape
    err = err.reshape(b,-1)
    err = torch.sum(err,dim=1)

    return err.tolist()

def compute_pred_error(pred,gt,opt,gt_track=True):
    # rec, gt: [B,N,2]

    if opt.headless:
        
        gt_pose = gt[:,-14:,:]
        pred_pose = pred[:,-14:,:]
        gt_boxs = torch.cat([gt[:,::16,:],gt[:,1::16,:]],dim=-1)
        pred_boxs = torch.cat([pred[:,::16,:],pred[:,1::16,:]],dim=-1)
        #gt_box = gt[:,-16:-14,:]
        # if gt_track:
        #     pred_box = gt[:,-16:-14,:]
        # else:
        #     pred_box = pred[:,-16:-14,:]
    else:
        pred_pose = pred[:,-17:,:]
        gt_pose = gt[:,-17:,:]
        gt_boxs = torch.cat([gt[:,::19,:],gt[:,1::19,:]],dim=-1)
        pred_boxs = torch.cat([pred[:,::19,:],pred[:,1::19,:]],dim=-1)
        #gt_box = gt[:,-19:-17,:]
        # if gt_track:
        #     pred_box = gt[:,-19:-17,:]
        # else:
        #     pred_box = pred[:,-19:-17,:]

    gt_pose_recover = recover_pose(gt_pose,gt_boxs,opt)
    
    if gt_track:
        predict_pose_recover = recover_pose(pred_pose,gt_boxs,opt)
    else:
        predict_pose_recover = recover_pose(pred_pose,pred_boxs,opt)

    err_l1 = torch.norm((gt_pose_recover - predict_pose_recover), p=1, dim=-1)
    err_l2 = torch.norm((gt_pose_recover - predict_pose_recover), p=2, dim=-1)
    # err_l2 (b,n)
    # err_l1 (b,n)
    #b,n = err_l1.shape
    err_l1 = err_l1.mean(dim=-1)
    err_l2 = err_l2.mean(dim=-1)

    return err_l1.tolist(), err_l2.tolist()

def normlize_score(scores):
    max_score = max(scores)
    min_score = min(scores)
    length = max_score - min_score
    scores = np.array(scores)

    return scores/length

def comput_auc(rec_err,meta,dataset,scene_norm=True):
    if dataset == 'ShanghaiTech':
        compute_dict = {}
        for err,name in zip(rec_err,meta):
            main, sub ,frame = name.split('_')
            
            scene = main + '_' + sub
            if scene not in compute_dict:
                compute_dict[scene] = {}
            if frame not in compute_dict[scene]:
                compute_dict[scene][frame] = [err]
            else:
                compute_dict[scene][frame].append(err)

        max_err_dict = {}
        all_label = []
        all_score = []

        for scene in compute_dict:
            max_err_dict[scene] = []
            frames = compute_dict[scene].keys()
            sorted_frames = list(sorted(frames))
            label = np.load(SHT_LABEL+scene+'.npy').tolist()
            all_label.extend(label)
            num_frame = len(label)
            anchor = 0

            for i in range(num_frame):
                
                if i >= len(sorted_frames):
                    max_err_dict[scene].append(0)

                else:
                    if int(sorted_frames[anchor]) == i:
                        max_rec = max(compute_dict[scene][sorted_frames[anchor]])
                        max_err_dict[scene].append(max_rec)
                        anchor += 1
                    else:
                        max_err_dict[scene].append(0)
                        
            ano_score = max_err_dict[scene]

            if scene_norm:
                ano_score = normlize_score(ano_score)

            all_score.extend(ano_score)

        all_score = gaussian_filter1d(all_score, 20)
        AUC = roc_auc_score(all_label, all_score)
            
        #fpr, tpr, thresholds = roc_curve(all_label, all_score, pos_label= 1)
        #AUC = auc(fpr,tpr)

        return AUC

def comput_auc_pred(rec_err,meta,dataset,opt,scene_norm=False):
    if dataset == 'ShanghaiTech':
        compute_dict = {}
        for err,name in zip(rec_err,meta):
            main, sub ,frame = name.split('_')
            
            scene = main + '_' + sub
            if scene not in compute_dict:
                compute_dict[scene] = {}
            if frame not in compute_dict[scene]:
                compute_dict[scene][frame] = [err]
            else:
                compute_dict[scene][frame].append(err)

        max_err_dict = {}
        all_label = []
        all_score = []
        
        for scene in compute_dict:
            max_err_dict[scene] = []
            frames = compute_dict[scene].keys()
            sorted_frames = list(sorted(frames))
            label = np.load(SHT_LABEL+scene+'.npy').tolist()
            all_label.extend(label[opt.track_length:])
            num_frame = len(label)
            anchor = 0

            for i in range(num_frame):
                
                if (i+1) > int(sorted_frames[-1]):
                    max_err_dict[scene].append(0)

                else:
                    
                    if int(sorted_frames[anchor]) == i:
                        max_rec = max(compute_dict[scene][sorted_frames[anchor]])
                        max_err_dict[scene].append(max_rec)
                        anchor += 1
                        
                    else:
                        max_err_dict[scene].append(0)
                        
            ano_score = max_err_dict[scene]
            
            if scene_norm:
                ano_score = normlize_score(ano_score)

            all_score.extend(ano_score[opt.track_length:])

        all_score = gaussian_filter1d(all_score, 20)
        AUC = roc_auc_score(all_label, all_score)

        #fpr, tpr, thresholds = roc_curve(all_label, all_score, pos_label= 1)
        #AUC = auc(fpr,tpr)

        return AUC

def get_kpts_img(kpts,remove_head=True):

    if not remove_head:
        links = [(0, 1), (0, 2), (1, 3), (2, 4),
                       (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 13), (13, 15), (12, 14), (14, 16),
                       (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    
        
    else:
        
        links = [      (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 13), (13, 15), (12, 14), (14, 16),
                       (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    kpts = np.array(kpts)

    x_ = kpts[:,0]
    y_ = kpts[:,1]
            
    w_ = np.max(x_) - np.min(x_)
    h_ = np.max(y_) - np.min(y_)

    scale_x = 100

    scale_y = 100*h_/w_

    img = np.zeros((100,100,3),np.uint8)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(links) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(links)):

        if remove_head:
            order1, order2 = links[i][0]-3, links[i][1]-3
        else:
            order1, order2 = links[i][0], links[i][1]

        x1 =int(((np.float32(x_[order1])))*scale_x)+50
        y1 =int(((np.float32(y_[order1])))*scale_y)+50
        x2 =int(((np.float32(x_[order2])))*scale_x)+50
        y2 =int(((np.float32(y_[order2])))*scale_y)+50
        cv2.line(img,(x1,y1),(x2,y2),thickness=1,color=colors[i])

    #img2 = img[:,:,::-1]     # transform image to rgb
    #plt.imshow(im2)
    #plt.show()

    return img


def get_img(kpts,remove_head=True):

    if not remove_head:
        links = [(0, 1), (0, 2), (1, 3), (2, 4),
                       (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 13), (13, 15), (12, 14), (14, 16),
                       (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    
        
    else:
        
        links = [      (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 13), (13, 15), (12, 14), (14, 16),
                       (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]

    x_ = kpts[2:,0]
    y_ = kpts[2:,1]
            
    w_ = np.max(x_) - np.min(x_)
    h_ = np.max(y_) - np.min(y_)

    scale_x = 100

    scale_y = 100*h_/w_

    img = np.zeros((856,480,3),np.uint8)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(links) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(links)):

        if remove_head:
            order1, order2 = links[i][0]-3, links[i][1]-3
        else:
            order1, order2 = links[i][0], links[i][1]

        x1 =int(((np.float32(x_[order1])))*scale_x)+50
        y1 =int(((np.float32(y_[order1])))*scale_y)+50
        x2 =int(((np.float32(x_[order2])))*scale_x)+50
        y2 =int(((np.float32(y_[order2])))*scale_y)+50
        cv2.line(img,(x1,y1),(x2,y2),thickness=1,color=colors[i])

    #img2 = img[:,:,::-1]     # transform image to rgb
    #plt.imshow(im2)
    #plt.show()

    return img

def get_whole_img(gt,pred,remove_head=True):

    if not remove_head:
        links = [(0, 1), (0, 2), (1, 3), (2, 4),
                       (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 13), (13, 15), (12, 14), (14, 16),
                       (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    
        
    else:
        
        links = [      (5, 7), (7, 9), (6, 8), (8, 10),
                       (11, 13), (13, 15), (12, 14), (14, 16),
                       (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]

    img = np.zeros((480,856,3),np.uint8)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(links) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(links)):

        if remove_head:
            order1, order2 = links[i][0]-3, links[i][1]-3
        else:
            order1, order2 = links[i][0], links[i][1]

        x1 =int(((np.float32(gt[order1][0]))))
        y1 =int(((np.float32(gt[order1][1]))))
        x2 =int(((np.float32(gt[order2][0]))))
        y2 =int(((np.float32(gt[order2][1]))))
        cv2.line(img,(x1,y1),(x2,y2),thickness=1,color=(0,0,255))

        x1_ =int(((np.float32(pred[order1][0]))))
        y1_ =int(((np.float32(pred[order1][1]))))
        x2_ =int(((np.float32(pred[order2][0]))))
        y2_ =int(((np.float32(pred[order2][1]))))

        cv2.line(img,(x1_,y1_),(x2_,y2_),thickness=1,color=(0,255,0))

    #img2 = img[:,:,::-1]     # transform image to rgb
    #plt.imshow(im2)
    #plt.show()

    return img

def save_samples(label, rec_joints, meta, iteration, sampledir,opt,show_num=2):

    for i in range(show_num):
        #print('gt_pose',label[i])
        #print('rec_pose',rec_joints[i])
        gt_pose = get_kpts_img(label[i],opt.headless)
        rec_pose = get_kpts_img(rec_joints[i],opt.headless)
        name = meta[i]
        combine = np.hstack((gt_pose,rec_pose))
        cv2.imwrite(sampledir+'/'+name+'_'+str(iteration)+'.jpg', combine)

def save_predict_samples(gt, pred, meta, iteration, sampledir,opt):
    # if opt.headless:
    #     label = np.array(label)[0,-16:,:]
    #     rec_joints = np.array(rec_joints)[0,-16:,:]
    # else:
    #     label = np.array(label)[0,-19:,:]
    #     rec_joints = np.array(rec_joints)[0,-19:,:]
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)

    if opt.headless:
        pred_pose = pred[:,-14:,:]
        pred_box = gt[:,-16:-14,:]
        gt_pose = gt[:,-14:,:]
        gt_box = gt[:,-16:-14,:]
    else:
        pred_pose = pred[:,-17:,:]
        pred_box = gt[:,-19:-17,:]
        gt_pose = gt[:,-17:,:]
        gt_box = gt[:,-19:-17,:]

    gt_pose_recover = recover_pose(gt_pose,gt_box)
    predict_pose_recover = recover_pose(pred_pose,pred_box)
    gt_pose_recover = gt_pose_recover.tolist()[0]
    predict_pose_recover = predict_pose_recover.tolist()[0]

    #gt_pose = get_img(label,opt.headless)
    #rec_pose = get_img(rec_joints,opt.headless)
    name = meta[0]
    combine = get_whole_img(predict_pose_recover,gt_pose_recover,opt.headless)
    #combine = np.hstack((gt_pose,rec_pose))
    cv2.imwrite(sampledir+'/'+name+'_'+str(iteration)+'.jpg', combine)

def save_parameters(flowgen):
    '''Write parameters setting file'''
    with open(os.path.join(flowgen.parameterdir, 'params.txt'), 'w') as file:
        file.write(flowgen.jobname)
        file.write('Training Parameters: \n')
        file.write(str(flowgen.opt) + '\n')
        if flowgen.load:
            file.write('Load pretrained model: ' + str(flowgen.load) + '\n')
            file.write('Iteration to load:' + str(flowgen.iter_to_load) + '\n')


if __name__ == '__main__':
    dummy_gt = torch.randint(1,10,(2,64,2)).tolist()
    dummy_pred = torch.randint(1,10,(2,64,2)).tolist()
    save_predict_samples(dummy_gt,dummy_pred,['1','2'],1,'./')

    # err = compute_pred_error(dummy_gt,dummy_pred)
    # print(err)

    #dummy_score = torch.rand(100000).tolist()
    #meta = json.load(open('./pose_meta_test.json'))[:100000]
    #auc = comput_auc_pred(dummy_score,meta,'ShanghaiTech')
    #print(auc)








