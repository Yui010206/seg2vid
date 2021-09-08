import os
import logging
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import sys 
sys.path.append(".") 
from utils.load_save import load_json, write_json
from utils.normalize import normalize_score,normalize_pose

POSE_META_FILE = 'pose_meta_{}_length{}_stride{}.json'
POSE_DATA_FILE = 'pose_data_{}_length{}_stride{}.json'

class ShanghaiTech(Dataset):
    def __init__(self, pose_dir, split='train', tracklet_len=8 , stride=2, head_less=False,
        normalize_tracklet=True, normalize_score=True, normalize_pose=True):

        self.pose_dir = pose_dir
        self.split = split
        self.head_less = head_less
        self.tracklet_len = tracklet_len
        self.stride = stride
        self.frame_width = 856
        self.frame_height = 480
        self.scale_factor = 100

        if self.head_less:
            self.joints_num =14
        else:
            self.joints_num =17
        self.type_token, self.spatial_token, self.temporal_token = self._gen_fixed_token_seq()
        self.meta_path = pose_dir + '/' + POSE_META_FILE.format(self.split,str(self.tracklet_len),str(self.stride))
        self.tracklet_path = pose_dir + '/' + POSE_DATA_FILE.format(self.split,str(self.tracklet_len),str(self.stride))

        self.normalize_tracklet = normalize_tracklet
        self.normalize_score = normalize_score
        self.normalize_pose = normalize_pose

        self._load_tracklets()

    def __len__(self):
        return len(self.meta_data)

    def _gen_fixed_token_seq(self):

        type_token = []
        spatial_token = []
        temporal_token = []
        single_type_tok = [0,0] + [1 for n in range(self.joints_num)]

        for i in range(self.tracklet_len):
            type_token.extend(single_type_tok)
            for j in range(self.joints_num+2):
                spatial_token.append(j)
                temporal_token.append(i+1)

        return torch.tensor(type_token), torch.tensor(spatial_token), torch.tensor(temporal_token)

    def _load_tracklets(self):

        if os.path.exists(self.tracklet_path) and os.path.exists(self.meta_path):
            logging.info('Load Traclets from saved files, Traclet Length {}, Stride {}'.format(self.tracklet_len,self.stride))
            self.meta_data, self.tracklet_data = self._lazy_load_tracklets()
        else:
            logging.info('Load Traclets from scratch, Traclet Length {}, Stride {}'.format(self.tracklet_len,self.stride))
            self.meta_data, self.tracklet_data = self._scratch_load_tracklets()

    def _lazy_load_tracklets(self):

        return load_json(self.meta_path), load_json(self.tracklet_path)

    def _scratch_load_tracklets(self):

        meta_data = []
        tracklet_data = []
        base_dir = self.pose_dir+'/'+self.split+'/tracked_person/'
        all_json = os.listdir(base_dir)
        logging.info('Processing raw traclets')
        filter_less_than = self.tracklet_len * self.stride + 4

        for file in tqdm(all_json):
            scene_tracks = load_json(base_dir+file)
            person_num = len(scene_tracks.keys())
            for p in scene_tracks.keys():
                tracks = scene_tracks[p]
                frame_num = len(tracks.keys())
                if frame_num < filter_less_than:
                    continue
                frame_index = list(sorted(tracks.keys()))
                for i in range(len(frame_index)-self.tracklet_len*self.stride):
                    select_frame = frame_index[i : i+self.tracklet_len*self.stride : self.stride]
                    simple_pose = [ np.around(np.array(tracks[f]['keypoints']),2).tolist() for f in select_frame ]
                    meta_data.append(file.split('_')[0]+'_'+file.split('_')[1]+'_'+select_frame[-1])
                    tracklet_data.append(simple_pose)

        logging.info('Process Done. Sample amount: ', len(meta_data))
        write_json(meta_data,self.meta_path)
        logging.info('Save meta data Done')
        write_json(tracklet_data,self.tracklet_path)
        logging.info('Save data Done')

        return meta_data,tracklet_data

    def _extract_boxes(self,tracklet,normalize=True):

        if normalize:
            box_xy_max = [[max(pose[::3])/self.frame_width,max(pose[1::3])/self.frame_height] for pose in tracklet]
            box_xy_min = [[min(pose[::3])/self.frame_width,min(pose[1::3])/self.frame_height] for pose in tracklet]
        else:
            box_xy_max = [[max(pose[::3]),max(pose[1::3])] for pose in tracklet]
            box_xy_min = [[min(pose[::3]),min(pose[1::3])] for pose in tracklet]

        return box_xy_max , box_xy_min

    def _extract_conf_score(self,tracklet,normalize=True):

        scores = []
        for pose in tracklet:
            pose_score = np.array(pose[2::3])
            if normalize:
                pose_score = normalize_score(pose_score)
            scores.append(pose_score.tolist())

        return scores

    def _extract_poses(self,tracklet,normalize=True):

        if isinstance(tracklet,list):
            tracklet = np.array(tracklet)
        x = tracklet[:, ::3]
        y = tracklet[:, 1::3]

        if normalize:
            x, y = normalize_pose(x,y)

        if isinstance(x,list):
            x, y = np.array(x), np.array(y)

        x = np.expand_dims(x,-1)
        y = np.expand_dims(y,-1)
        pose = np.concatenate((x,y),axis=-1).tolist()

        return pose

    def _flat_input(self,poses, boxes_max, boxes_min, scores):

        assert len(poses) == len(boxes_max)
        assert len(boxes_max) == len(boxes_min)
        assert len(poses) == len(scores)

        kpts = []
        weights = []

        for i in range(len(poses)):
            kpts.append(boxes_min[i])
            kpts.append(boxes_max[i])
            kpts.extend(poses[i])

            weights.append(1)
            weights.append(1)
            weights.extend(scores[i])
            
        return kpts, weights

    def __getitem__(self, idx):

        meta = self.meta_data[idx]
        tracklet = self.tracklet_data[idx]

        boxes_max, boxes_min = self._extract_boxes(tracklet,self.normalize_tracklet)
        scores = self._extract_conf_score(tracklet,self.normalize_score)
        poses = self._extract_poses(tracklet,self.normalize_pose)
        gt_seqence, weights = self._flat_input(poses, boxes_max, boxes_min,scores)

        gt_seqence = torch.tensor(gt_seqence)*self.scale_factor
        weights = torch.tensor(weights)

        type_token = self.type_token.clone()
        spatial_token = self.spatial_token.clone()
        temporal_token = self.temporal_token.clone()

        mask = torch.tensor([1 for i in range((self.tracklet_len-1)*(self.joints_num+2))] + [0 for i in range(self.joints_num+2)])
        mask_ = torch.cat((mask.unsqueeze(0),mask.unsqueeze(0)),dim=0).permute(1,0)
        mask_index = mask_==0

        input_sequence = copy.deepcopy(gt_seqence)
        input_sequence[mask_index] = -100

        input_dict = {
            'meta': meta,
            'gt_sequence':gt_seqence,
            'input_sequence': input_sequence,
            'weigths': weights,
            'type_token': type_token,
            'spatial_token':spatial_token,
            'temporal_token':temporal_token,
            'frame_width':self.frame_width,
            'frame_height':self.frame_height,
            'scale_factor': self.scale_factor,
            'joints_num':self.joints_num
            }

        return input_dict

if __name__ == '__main__':

    from dataset_path import *
    import cv2
    from torch.utils.data import DataLoader
    sys.path.append(".") 
    from utils.visualization import visualize_local_tracklets, visualize_global_tracklets
    
    debug_Dataset = ShanghaiTech(pose_dir=ShanghaiTech_Pose_Dir,split='test',tracklet_len=8 , stride=2, head_less=False)
    dataloader = DataLoader(debug_Dataset, batch_size=2, shuffle=True, num_workers=0)
    VIS = True

    for i, input_dict in enumerate(tqdm(dataloader)):

        print(i)
        if VIS:
            track_img = visualize_local_tracklets(input_dict['gt_sequence'][0],input_dict['joints_num'][0],input_dict['scale_factor'][0])
            #track_img = visualize_global_tracklets(input_dict['gt_sequence'][0],input_dict['joints_num'][0],
            #    input_dict['scale_factor'][0],input_dict['frame_width'][0],input_dict['frame_height'][0])
            cv2.imwrite('./{}.jpg'.format(str(i)),track_img)

        if i>10:
            break


