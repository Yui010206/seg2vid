import torch
import torch.nn as nn
import numpy as np


class OKS(nn.Module):
    def __init__(self):
        super(OKS, self).__init__()
        #self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_pose_, target_pose_,weight_):
        # predicted: B,N,2
        # mask: B, N
        # weitgt: B, N
        predicted_pose = predicted_pose_.clone()
        target_pose = target_pose_.clone()
        weight = weight_.clone()

        assert predicted_pose.shape == target_pose.shape

        norm_pose = torch.norm((predicted_pose - target_pose), p=2, dim=-1)*weight
        loss = norm_pose.mean()

        return loss 

class IOU(nn.Module):
    def __init__(self):
        super(IOU, self).__init__()
        
    def forward(self, predicted_, target_):
        # predicted: B,N,2
        # mask: B, N
        predicted = predicted_.clone()
        target = target_.clone()

        pre_x_min = predicted[:,0,0]
        pre_x_max = predicted[:,1,0]
        pre_y_min = predicted[:,0,1]
        pre_y_max = predicted[:,1,1]

        gt_x_min = target[:,0,0]
        gt_x_max = target[:,1,0]
        gt_y_min = target[:,0,1]
        gt_y_max = target[:,1,1]

        gt_box = torch.cat([gt_x_min.unsqueeze(1),gt_y_min.unsqueeze(1),gt_x_max.unsqueeze(1),gt_y_max.unsqueeze(1)],dim=1)
        pred_box = torch.cat([pre_x_min.unsqueeze(1),pre_y_min.unsqueeze(1),pre_x_max.unsqueeze(1),pre_y_max.unsqueeze(1)],dim=1)

        #iou = bbox_iou(gt_box,pred_box,True)
        giou = bbox_giou(gt_box,pred_box)

        loss = 1-giou.mean()

        return loss


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
        # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    #print(b1_x1)
    #print(b2_x1)
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    #print(inter_rect_x1)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def bbox_giou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """


        # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    #print(b1_x1)
    #print(b2_x1)
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    #print(inter_rect_x1)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

    expand_rect_x1 = torch.min(b1_x1, b2_x1)
    expand_rect_y1 = torch.min(b1_y1, b2_y1)
    expand_rect_x2 = torch.max(b1_x2, b2_x2)
    expand_rect_y2 = torch.max(b1_y2, b2_y2)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    expand_area = (expand_rect_x2 - expand_rect_x1) * (expand_rect_y2 - expand_rect_y1)

    giou = iou - ((expand_area - (b1_area + b2_area - inter_area + 1e-16))/expand_area)

    return giou

if __name__ == '__main__':
    loss = IOU_MASK()

    mask = torch.tensor([[0,0,0,1],[1,0,0,0]]).bool()
    predicted = torch.randint(1,10,(4,32,2))
    target = torch.randint(1,10,(4,32,2))
    mask = torch.randint(0,2,(4,32))
    mask[0] = 1
    mask[1] = 1

    l = loss(predicted,target,mask,True)
    print(l)