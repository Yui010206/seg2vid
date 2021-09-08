import os, time, sys
import torch

from torch.utils.data import DataLoader
from dataset import get_training_set, get_test_set
from tqdm import tqdm
from opts import parse_opts

from models.APF import AnoPoseFormer
from utils.

import torch.optim as optim

import logging

def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

class AnoPose(object):

    def __init__(self, opt):
        self.opt = opt

        dataset = opt.dataset

        self.dataset_name = dataset

        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        self.jobname = dataset + opt.env#'_gpu8_256_1node' 

        self.modeldir = self.jobname + 'model'
        self.sampledir = os.path.join(self.workspace, self.jobname)
        self.parameterdir = self.sampledir + '/params'
        self.log_path = self.sampledir + '/log'
        self.events_root = self.sampledir + '/events'


        if not os.path.exists(self.parameterdir):
            os.makedirs(self.parameterdir)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # whether to start training from an existing snapshot
        self.load = False

        self.iter_to_load = opt.iter_to_load

        # Write parameters setting file
        if os.path.exists(self.parameterdir):
            utils.save_parameters(self)

        if not opt.inference:
            train_Dataset = get_training_set(opt)
            self.trainloader = DataLoader(train_Dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                      pin_memory=True, drop_last=True)


        test_Dataset = get_test_set(opt)
        self.testloader = DataLoader(test_Dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                                     pin_memory=True, drop_last=False)

    def train(self):

        opt = self.opt
        gpu_ids = range(torch.cuda.device_count())
        logger.info('Number of GPUs in use {}'.format(gpu_ids))
        iteration = 0

        APF = AnoPoseFormer(num_frame=opt.track_length,headless=opt.headless,embed_dim=opt.embed_dim, depth=opt.depth).cuda()
    
        if torch.cuda.device_count() > 1:
            APF = nn.DataParallel(APF).cuda()

        oks_loss = losses.OKS()
        iou_loss = losses.IOU()

        logger.info(self.jobname)

        optimizer = optim.AdamW(APF.parameters(), lr=opt.lr_rate, eps = 1e-8)

        if self.load:

            model_name = self.sampledir + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
            logger.info("loading model from {}".format(model_name))

            state_dict = torch.load(model_name)
            if torch.cuda.device_count() > 1:
                APF.module.load_state_dict(state_dict['APF'])
                optimizer.load_state_dict(state_dict['optimizer'])
            else:
                APF.load_state_dict(state_dict['APF'])
                optimizer.load_state_dict(state_dict['optimizer'])
                
            iteration = self.iter_to_load + 1

        for epoch in range(opt.num_epochs):

            logger.info('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
            logger.info('-' * 10)

            for input_dict in iter(self.trainloader):
                # get the inputs
                input = input_dict['input_sequence'].float().cuda()
                label = input_dict['gt_sequence'].float().cuda()
                weigth = input_dict['weigth'].float().cuda()
                type_token = input_dict['type_token'].long().cuda()
                spatial_token = input_dict['spatial_token'].long().cuda()
                temporal_token = input_dict['temporal_token'].long().cuda()
                meta = input_dict['meta']

                start = time.time()
                # Set train mode
                APF.train()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize

                output = APF(input,type_token,spatial_token,temporal_token,opt.train_mode,opt.headless)
                
                loss = 0

                if 'MPR' in opt.train_mode:

                    loss += mpjpe_loss(recons_joints,label,mask_mpr)

                if 'MPR' in opt.train_mode:

                if 'MPR' in opt.train_mode:

                if 'MPR' in opt.train_mode:
                    
                # backward
                loss.backward()
                # Update
                optimizer.step()
                end = time.time()

                # print statistics
                # iteration!=0 and 
                if iteration % 20 == 0:
                    logger.info("iter {} (epoch {}), mpm_loss = {:.6f}, time/batch = {:.3f}"
                        .format(iteration, epoch, loss.item(), end - start))

                #if iteration % 200 == 0:
                #    utils.save_predict_samples(label.cpu().tolist(), predicted_joints.cpu().tolist(), meta, iteration, self.sampledir, opt)

                if iteration!=0 and iteration % 1000 == 0:
                    logger.info('Start evaluation!')
                        # Set to evaluation mode (randomly sample z from the whole distribution)
                    all_rec_err_l1 = []
                    all_rec_err_l2 = []
                    all_pre_err_l1 = []
                    all_pre_err_l2 = []
                    all_meta = []

                    with torch.no_grad():
                        APF.eval()
                        for i,input_dict in enumerate(tqdm(self.testloader)):
                            #for input_dict in iter(self.testloader):
                            meta = input_dict['meta']

                            if opt.test_mode == 'MPR':

                                input_mpr = input_dict['gt_sequence'].clone().float().cuda()
                                label = input_dict['gt_sequence'].float().cuda()
                                type_token = input_dict['type_token'].long().cuda()
                                spatial_token = input_dict['spatial_token'].long().cuda()
                                temporal_token = input_dict['temporal_token'].long().cuda()
                                recons_joints = APF(input_mpr,type_token,spatial_token,temporal_token,opt.test_mode,opt.headless)

                                pred_error = utils.compute_rec_error(recons_joints.cpu(),label.cpu(),opt,gt_track=True)

                                all_meta.extend(meta)
                                all_err.extend(pred_error)

                            elif opt.test_mode == 'MPP':

                                input_mpp = input_dict['input_mpp'].clone().float().cuda()
                                label = input_dict['gt_sequence'].float().cuda()
                                type_token = input_dict['type_token'].long().cuda()
                                spatial_token = input_dict['spatial_token'].long().cuda()
                                temporal_token = input_dict['temporal_token'].long().cuda()
                                pred_joints = APF(input_mpp,type_token,spatial_token,temporal_token,opt.test_mode,opt.headless)

                                pred_error_l1, pred_error_l2 = utils.compute_pred_error(pred_joints.cpu(),label.cpu(),opt,gt_track=True)

                                all_meta.extend(meta)
                                all_err_l1.extend(pred_error_l1)
                                all_err_l2.extend(pred_error_l2)


                            elif opt.test_mode == 'MTR':

                                pass

                            elif opt.test_mode == 'MTP':
                                pass

                            elif opt.test_mode == 'Rec':
                                pass
                            
                            elif opt.test_mode == 'Pred':

                                pred_pose = APF(input_mpp,type_token,spatial_token,temporal_token,'MPP',opt.headless)
                                # B,N,2
                                pred_track = APF(input_mtp,type_token,spatial_token,temporal_token,opt.train_mode,opt.headless)
                                predict = torch.cat([pred_track,pred_pose],dim=1)
                                pred_error_l1, pred_error_l2 = utils.compute_pred_error(predict.cpu(),label.cpu(),opt,gt_track=False)
                                all_meta.extend(meta)
                                all_err_l1.extend(pred_error_l1)
                                all_err_l2.extend(pred_error_l2)
                                
                            elif opt.test_mode == 'All':
                                pass

                    utils.save_json(all_err_l1,iteration, 'err_l1',self.sampledir)
                    utils.save_json(all_err_l2,iteration, 'err_l2',self.sampledir)
                    utils.save_json(all_meta,iteration, 'meta',self.sampledir)

                    auc_l1 = utils.comput_auc_pred(all_err_l1,all_meta,self.dataset_name,opt,scene_norm=False)
                    auc_l2 = utils.comput_auc_pred(all_err_l2,all_meta,self.dataset_name,opt,scene_norm=False)
                    auc_l1_nor = utils.comput_auc_pred(all_err_l1,all_meta,self.dataset_name,opt,scene_norm=True)
                    auc_l2_nor = utils.comput_auc_pred(all_err_l2,all_meta,self.dataset_name,opt,scene_norm=True)

                    logger.info('L1 Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l1))
                    logger.info('L2 Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l2))
                    logger.info('L1 Norm Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l1_nor))
                    logger.info('L2 Norm Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l2_nor))

                        # Save model's parameter
                    checkpoint_path = self.sampledir + '/{:06d}_model.pth.tar'.format(iteration)

                    logger.info("model saved to {}".format(checkpoint_path))

                    if torch.cuda.device_count() > 1:
                        torch.save({'APF': APF.state_dict(), 'optimizer': optimizer.state_dict()},
                                       checkpoint_path)
                    else:
                        try:
                            torch.save({'APF': APF.module.state_dict(), 'optimizer': optimizer.state_dict()},
                                           checkpoint_path)
                        except:
                            torch.save({'APF': APF.state_dict(), 'optimizer': optimizer.state_dict()},
                                       checkpoint_path)

                iteration += 1

    def test(self):

        model_name = self.sampledir + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
        logger.info("loading model from {}".format(model_name))
        state_dict = torch.load(model_name)
        APF = AnoPoseFormer(num_frame=opt.track_length,headless=opt.headless,embed_dim=opt.embed_dim, depth=opt.depth).cuda()
        try:
            APF.module.load_state_dict(state_dict['APF'])
        except:
            APF.load_state_dict(state_dict['APF'])

        logger.info('Start evaluation!')
                        # Set to evaluation mode (randomly sample z from the whole distribution)
        all_err_l1 = []
        all_err_l2 = []
        all_meta = []
        all_recons_joints = []

        with torch.no_grad():
            APF.eval()
            for i,input_dict in enumerate(tqdm(self.testloader)):
                            #for input_dict in iter(self.testloader):
                meta = input_dict['meta']

                if opt.test_mode == 'MPR':

                    input_mpr = input_dict['gt_sequence'].clone().float().cuda()
                    label = input_dict['gt_sequence'].float().cuda()
                    type_token = input_dict['type_token'].long().cuda()
                    spatial_token = input_dict['spatial_token'].long().cuda()
                    temporal_token = input_dict['temporal_token'].long().cuda()
                    recons_joints = APF(input_mpr,type_token,spatial_token,temporal_token,opt.test_mode,opt.headless)

                    pred_error = utils.compute_rec_error(recons_joints.cpu(),label.cpu(),opt,gt_track=True)

                    all_meta.extend(meta)
                    all_err.extend(pred_error)

                elif opt.test_mode == 'MPP':

                    input_mpp = input_dict['input_mpp'].clone().float().cuda()
                    label = input_dict['gt_sequence'].float().cuda()
                    type_token = input_dict['type_token'].long().cuda()
                    spatial_token = input_dict['spatial_token'].long().cuda()
                    temporal_token = input_dict['temporal_token'].long().cuda()
                    pred_joints = APF(input_mpp,type_token,spatial_token,temporal_token,opt.test_mode,opt.headless)

                    pred_error_l1, pred_error_l2 = utils.compute_pred_error(pred_joints.cpu(),label.cpu(),opt,gt_track=True)

                    all_meta.extend(meta)
                    all_err_l1.extend(pred_error_l1)
                    all_err_l2.extend(pred_error_l2)


                elif opt.test_mode == 'MTR':

                    pass

                elif opt.test_mode == 'MTP':
                    pass

                elif opt.test_mode == 'Rec':
                    pass
                                
                elif opt.test_mode == 'Pred':
                    pass

                elif opt.test_mode == 'All':
                    pass

            utils.save_json(all_err_l1,-1, 'err_l1',self.sampledir)
            utils.save_json(all_err_l2,-1, 'err_l2',self.sampledir)
            utils.save_json(all_meta,-1, 'meta',self.sampledir)

            auc_l1 = utils.comput_auc_pred(all_err_l1,all_meta,self.dataset_name,opt,scene_norm=False)
            auc_l2 = utils.comput_auc_pred(all_err_l2,all_meta,self.dataset_name,opt,scene_norm=False)
            auc_l1_nor = utils.comput_auc_pred(all_err_l1,all_meta,self.dataset_name,opt,scene_norm=True)
            auc_l2_nor = utils.comput_auc_pred(all_err_l2,all_meta,self.dataset_name,opt,scene_norm=True)

            logger.info('L1 Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l1))
            logger.info('L2 Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l2))
            logger.info('L1 Norm Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l1_nor))
            logger.info('L2 Norm Anomaly Score AUC on {}: {}'.format(self.dataset_name,auc_l2_nor))
                            # Save model's parameter
            logger.info('====================Evaluation End====================')


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    opt = parse_opts()
    print (opt)
    '''Dist Init!!'''
    pipeline = AnoPose(opt)
    if opt.inference:
        pipeline.test()
    else:
        pipeline.train()
