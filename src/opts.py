import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=512,
        type=int,
        help='batch size')

    parser.add_argument(
        '--MPR',
        default=0.2,
        type=float,
        help='turn on MPR task')

    parser.add_argument(
        '--MPP',
        default=0.3,
        type=float,
        help='turn on MPR task')

    parser.add_argument(
        '--MTR',
        default=0.2,
        type=float,
        help='turn on MPR task')

    parser.add_argument(
        '--MTP',
        default=0.3,
        type=float,
        help='turn on MPR task')

    parser.add_argument(
        '--test_mode',
        default='MPP',
        type=str,
        help='test mode')

    parser.add_argument(
        '--train_mode',
        default='MPP',
        type=str,
        help='test mode')

    parser.add_argument(
        '--env',
        default='_gpu8_256_1node',
        type=str,
        help='weight of reconstruction loss.')

    parser.add_argument(
        '--embed_dim',
        default=128,
        type=int,
        help='load checkpoints')

    parser.add_argument(
        '--depth',
        default=8,
        type=int,
        help='load checkpoints')

    parser.add_argument(
        '--iter_to_load',
        default=5000,
        type=int,
        help='load checkpoints')

    parser.add_argument(
        '--inference',
        default=False,
        type=bool,
        help='turn on inference mode')

    parser.add_argument(
        '--cal_giou',
        default=False,
        type=bool,
        help='turn on inference mode')
    
    parser.add_argument('--local_rank', 
        type=int,default=0,help='node rank for distributed training')

    parser.add_argument(
        '--track_length',
        default=8,
        type=int,
        help='number of frames for each tracklets')
    parser.add_argument(
        '--stride',
        default=2,
        type=int,
        help='number of frames for each tracklets')

    parser.add_argument(
        '--mask_ratio',
        default=0.3,
        type=float,
        help='mask input ratio')

    parser.add_argument(
        '--headless',
        default=False,
        type=bool,
        help='use head joints or not')
    parser.add_argument(
        '--mean_box',
        default=False,
        type=bool,
        help='use mean box or not')
    parser.add_argument(
        '--coco_18',
        default=False,
        type=bool,
        help='use coco18 format or not')

    parser.add_argument(
        '--num_epochs',
        default=100,
        type=int,
        help=
        'Max. number of epochs to train.'
    )

    parser.add_argument(
        '--lr_rate',
        default=5e-5,
        type=float,
        help='learning rate used for training.'
    )

    parser.add_argument(
        '--workers',
        default=2,
        type=int,
        help='number of workers used for data loading.'
    )
    parser.add_argument(
        '--dataset',
        default='ShanghaiTech',
        type=str,
        help=
        'Used dataset (cityscpes | cityscapes_two_path | kth | ucf101).'
    )

    parser.add_argument(
        '--seed',
        default=31415,
        type=int,
        help='Manually set random seed'
    )


    args = parser.parse_args()

    return args