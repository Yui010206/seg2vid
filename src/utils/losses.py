import torch
import torch.nn as nn
import numpy as np


class OKS_Loss(nn.Module):
    def __init__(self):
        super(OKS_Loss, self).__init__()
        
    def forward(self, predicted_pose, target_pose, weight=None):
        # predicted: B,N,2
        # mask: B, N
        # weitgt: B, N

        assert predicted_pose.shape == target_pose.shape

        norm_pose = torch.norm((predicted_pose - target_pose), p=2, dim=-1)
        if weitgt is not None:
            norm_pose *= weight

        loss = norm_pose.mean()
        return loss 

class DIOU_Loss(nn.Module):
    def __init__(self):
        super(DIOU_Loss, self).__init__()
        
    def forward(self, predicted, target):
        # predicted: B,N,2
        # mask: B, N

        # predicted [B,N,2]
        # [B,N,4]

        b, n, _ = predicted.shape

        assert predicted.shape == target.shape

        pre_xy_min = predicted[:,::2,:]
        pre_xy_max = predicted[:,1::2,:]

        gt_xy_min = target[:,::2,:]
        gt_xy_max = target[:,1::2,:]

        pre_boxes = torch.cat([pre_xy_min,pre_xy_max],dim=-1).reshape(-1,4)
        gt_boxes = torch.cat([gt_xy_min,gt_xy_max],dim=-1).reshape(-1,4)

        loss = bboxes_diou(pre_boxes,gt_boxes)

        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 1.), loss)
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.), loss)

        #print(giou)

        # loss = 1 - giou.mean()
        return loss.mean()


# def bbox_giou(box1, box2):
#     """
#     Returns the IoU of two bounding boxes
#     """
#     # Get the coordinates of bounding boxes
#     b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#     b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

#     # get the coordinates of the intersection rectangle
#     #print(b1_x1)
#     #print(b2_x1)
#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)
#     #print(inter_rect_x1)
#     # Intersection area
#     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
#     # Union Area
#     b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
#     b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

#     expand_rect_x1 = torch.min(b1_x1, b2_x1)
#     expand_rect_y1 = torch.min(b1_y1, b2_y1)
#     expand_rect_x2 = torch.max(b1_x2, b2_x2)
#     expand_rect_y2 = torch.max(b1_y2, b2_y2)

#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

#     expand_area = (expand_rect_x2 - expand_rect_x1) * (expand_rect_y2 - expand_rect_y1)

#     giou = iou - ((expand_area - (b1_area + b2_area - inter_area + 1e-16))/expand_area)

#     return giou

def bboxes_diou(boxes1,boxes2):
    '''
    cal DIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    #cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    #cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])

    #cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
                 np.square(boxes1_center[...,1]-boxes2_center[...,1])

    #cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious

if __name__ == '__main__':
    loss = GIOU_Loss()

    predicted = torch.randint(1,10,(2,16,2))
    target = torch.randint(1,10,(2,16,2))

    l = loss(predicted,target)

    print(l)