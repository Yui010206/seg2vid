from dataset_path import *
import os

def get_training_set(opt):
    assert opt.dataset in ['ShanghaiTech', 'Avenue', 'IITB']

    if opt.dataset == 'ShanghaiTech':
        from ShanghaiTech import ShanghaiTech

        train_Dataset = ShanghaiTech(pose_dir=ShanghaiTech_Pose_Dir, split='train', tracklet_len=opt.tracklet_len ,stride=opt.stride,head_less=opt.headless)

    elif opt.dataset == 'IITB':
        pass

    elif opt.dataset == 'Avenue':

        pass

    else:

        raise ValueError ("Dataset Name Invalid!")



    return train_Dataset


def get_test_set(opt):
    assert opt.dataset in ['ShanghaiTech', 'Avenue', 'IITB']

    if opt.dataset == 'ShanghaiTech':
        from ShanghaiTech import ShanghaiTech

        test_Dataset = ShanghaiTech(pose_dir=ShanghaiTech_Pose_Dir,split='test', tracklet_len=opt.tracklet_len ,stride=opt.stride, head_less=opt.headless)

    elif opt.dataset == 'IITB':
        pass

    elif opt.dataset == 'Avenue':

        pass

    else:
        raise ValueError ("Dataset Name Invalid!")


    return test_Dataset
