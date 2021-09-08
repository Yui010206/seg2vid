import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def visualize_local_tracklets(flat_sequence_,joints_num,scale_factor):

    flat_sequence = flat_sequence_.clone()
    flat_sequence = flat_sequence.reshape((-1,int(joints_num+2),2))
    tracklet = []

    for pose in flat_sequence:
        tracklet.append(pose[2:,:].tolist())

    tracklet_img = visualize_tracklet(tracklet,scale_factor)

    return tracklet_img

def visualize_global_tracklets(flat_sequence_,joints_num,scale_factor,frame_width,frame_height):

    flat_sequence = flat_sequence_.clone()

    flat_sequence = flat_sequence.reshape((-1,int(joints_num+2),2))

    tracklet = []
    box_max = []
    box_min = []

    for pose in flat_sequence:
        tracklet.append(pose[2:,:])
        box_min.append(pose[0,:])
        box_max.append(pose[1,:])

    img = visualize_on_whole_frame(tracklet,box_min,box_max,scale_factor,frame_width,frame_height)

    return img

def visualize_on_whole_frame(tracklet,box_min,box_max,scale_factor,frame_width,frame_height):

    img = np.zeros((frame_height,frame_width,3),np.uint8)
    links = [(0, 1), (0, 2), (1, 3), (2, 4),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]

    for pose, b_min, b_max in zip(tracklet,box_min,box_max):

        x_min ,y_min = b_min[0]/scale_factor*frame_width, b_min[1]/scale_factor*frame_height
        x_max ,y_max = b_max[0]/scale_factor*frame_width, b_max[1]/scale_factor*frame_height
        w, h = x_max-x_min, y_max-y_min
        x_c , y_c = (x_max+x_min)/2, (y_max+y_min)/2

        x = ((pose[:,0]/scale_factor)*w + x_c).numpy()
        y = ((pose[:,1]/scale_factor)*h + y_c).numpy()
        for i in range(len(links)):
            order1, order2 = links[i][0], links[i][1]
            x1 =int(((np.float32(x[order1])))) + int(scale_factor/2)
            y1 =int(((np.float32(y[order1])))) + int(scale_factor/2)
            x2 =int(((np.float32(x[order2])))) + int(scale_factor/2)
            y2 =int(((np.float32(y[order2])))) + int(scale_factor/2)
            cv2.line(img,(x1,y1),(x2,y2),thickness=1,color=(0, 225, 0))

    return img


def visualize_tracklet(tracklet,scale_factor):

    imgs = []
    for pose in tracklet:
        img = visulize_single_pose(pose,scale_factor)
        imgs.append(img)
    imgs = np.hstack(imgs)

    return imgs

def visulize_single_pose(kpts,scale_factor):

    links = [(0, 1), (0, 2), (1, 3), (2, 4),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    
    kpts = np.array(kpts)

    x = kpts[:,0]
    y = kpts[:,1]

    img = np.zeros((100,100,3),np.uint8)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(links) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i in range(len(links)):

        order1, order2 = links[i][0], links[i][1]
        x1 =int(((np.float32(x[order1])))) + int(scale_factor/2)
        y1 =int(((np.float32(y[order1])))) + int(scale_factor/2)
        x2 =int(((np.float32(x[order2])))) + int(scale_factor/2)
        y2 =int(((np.float32(y[order2])))) + int(scale_factor/2)
        cv2.line(img,(x1,y1),(x2,y2),thickness=1,color=colors[i])

    return img
