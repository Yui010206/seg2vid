import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',default=2,type=int,help='batch size')
    parser.add_argument('--lr_rate',default=5e-5,type=float)
    parser.add_argument('--epochs',default=100,type=int)

    parser.add_argument('--seed',default=2021,type=int)

    parser.add_argument('--workers',default=2,type=int)

    parser.add_argument('--dataset',default='ShanghaiTech',type=str)

    parser.add_argument('--MPR',default=0.2,type=float,help='turn on MPR task')
    parser.add_argument('--MPP',default=0.3,type=float,help='turn on MPR task')
    parser.add_argument('--MTR',default=0.2,type=float,help='turn on MPR task')
    parser.add_argument('--MTP',default=0.3,type=float,help='turn on MPR task')

    parser.add_argument('--train_tasks',default='MPP',type=str)
    parser.add_argument('--test_tasks',default='MPP',type=str)
    parser.add_argument('--eval_interval',default=1,type=int)

    parser.add_argument('--load_pretrain_model',default=False,type=bool)
    parser.add_argument('--iter_to_load',default=5000,type=int,help='load checkpoints')
    parser.add_argument('--inference',default=False,type=bool,help='turn on inference mode')

    parser.add_argument('--embed_dim',default=128,type=int)
    parser.add_argument('--depth',default=8,type=int)

    parser.add_argument('--tracklet_len',default=8,type=int)
    parser.add_argument('--stride',default=2,type=int)
    parser.add_argument('--headless',default=False,type=bool)

    args = parser.parse_args()

    return args