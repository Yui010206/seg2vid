import os, time, sys
import torch


from torch.utils.data import DataLoader
from datasets.datasets import get_training_set, get_test_set
from tqdm import tqdm
from opts import parse_opts

from models.APF import AnoPoseFormer
from utils.logger import get_logger
from utils.load_save import save_parameters
from utils.losses import OKS_Loss, GIOU_Loss

from transformers import AdamW,get_linear_schedule_with_warmup

class Train_Eval_Inference(object):

    def __init__(self, opt):
        self.opt = opt
        self.dataset_name = opt.dataset

        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        self.jobname = dataset + '_' + opt.train_tasks
        self.exp_dir = os.path.join(self.workspace, self.jobname)
        self.model_save_dir = os.path.join(self.exp_dir, 'models')
        self.vis_sample_dir = os.path.join(self.exp_dir, 'vis_samples')

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.vis_sample_dir):
            os.makedirs(self.vis_sample_dir)

        # whether to start training from an existing snapshot
        self.load_pretrain_model = opt.load_pretrain_model
        if self.load_pretrain_model:
            self.iter_to_load = opt.iter_to_load

        save_parameters(self.exp_dir,opt)

        train_Dataset = get_training_set(opt)
        self.train_loader = DataLoader(train_Dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                      pin_memory=True, drop_last=True)
        test_Dataset = get_test_set(opt)
        self.test_loader = DataLoader(test_Dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                                     pin_memory=True, drop_last=False)

        self.logger = get_logger(self.exp_dir + '/log.txt')

        self.oks_loss = OKS_Loss()
        self.iou_loss = GIOU_Loss()

    def train_batch(self,model,optimizer, scheduler,epoch,iteration):

        for input_dict in iter(self.trainloader):

            input = input_dict['input_sequence'].float().cuda()
            weigths = input_dict['weigths'].float().cuda()
            type_token = input_dict['type_token'].long().cuda()
            spatial_token = input_dict['spatial_token'].long().cuda()
            temporal_token = input_dict['temporal_token'].long().cuda()
            meta = input_dict['meta']

            model.zero_grad()

            output = model(input,type_token,spatial_token,temporal_token)
                
            loss = 0
            mpr_loss = 0
            mpp_loss = 0
            mtr_loss = 0
            mtp_loss = 0

            if 'MPR' in self.train_tasks:
                mpr_loss = self.oks_loss(output['MPR_output'],)
                loss += mpr_loss

            if 'MPR' in self.train_task:

            if 'MPR' in self.train_task:

            if 'MPR' in self.train_task:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()


    def eval_batch(self):

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


    def inference(self):

        pass



    def train_eval(self):

        gpu_ids = range(torch.cuda.device_count())
        self.logger.info('Number of GPUs in use {}'.format(gpu_ids))
        model = AnoPoseFormer(tracklet_len=self.opt.tracklet_len,headless=self.opt.headless,embed_dim=self.opt.embed_dim, depth=self.opt.depth,tasks=self.opt.train_tasks).cuda()
        
        total_steps = len(self.train_loader) * epochs
        optimizer = AdamW(model.parameters(), lr=opt.lr_rate, eps = 1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                               num_warmup_steps = 80, # Default value in run_glue.py
                                               num_training_steps = total_steps)

        self.logger.info(self.jobname)

        iteration = 0
        if self.load_pretrain_model:
            model_name = self.model_save_dir + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
            logger.info("loading model from {}".format(model_name))
            state_dict = torch.load(model_name)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            iteration = self.iter_to_load + 1

        self.logger.info('Start Training!')

        for epoch in range(self.opt.epochs):

            model.train()
            total_train_loss,iteration = self.train_batch(model,optimizer, scheduler,epoch,iteration)
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.logger.info("epoch {}, total_train_loss: {:.4f}, avg_train_loss: {:.4f}")

            if epoch%self.opt.eval_interval==0:
                model.eval()
                total_eval_loss = self.eval_batch(model,epoch, self.opt.test_tasks)
                self.logger.info("epoch {}, total_val_loss: {:.4f}, avg_val_loss: {:.4f}")

        self.logger.info('End Training!')

if __name__ == '__main__':

    opt = parse_opts()
    print (opt)

    pipeline = Train_Eval_Inference(opt)

    if opt.inference:
        pipeline.inference()
    else:
        pipeline.train_eval()
