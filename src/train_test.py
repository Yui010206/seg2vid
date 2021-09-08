import os, time, sys
import torch


from torch.utils.data import DataLoader
from datasets.datasets import get_training_set, get_test_set
from tqdm import tqdm
from opts import parse_opts

from models.APF import AnoPoseFormer
from utils.logger import get_logger
from utils.load_save import save_parameters
from utils.losses import OKS_Loss, DIOU_Loss
from utils.visualization import visualize_local_tracklets, visualize_global_tracklets

from transformers import AdamW,get_linear_schedule_with_warmup

class Train_Eval_Inference(object):

    def __init__(self, opt):
        self.opt = opt
        self.dataset_name = opt.dataset

        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        self.jobname = opt.dataset + '_' + opt.train_tasks
        self.exp_dir = os.path.join(self.workspace, self.jobname)
        self.model_save_dir = os.path.join(self.exp_dir, 'models')
        self.vis_sample_dir = os.path.join(self.exp_dir, 'vis_samples')

        self.train_tasks = opt.train_tasks
        self.test_tasks = opt.test_tasks

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
        self.giou_loss = DIOU_Loss()

    def train_batch(self,model,optimizer, scheduler,epoch,iteration):

        total_loss = 0
        total_mpr_loss = 0
        total_mpp_loss = 0
        total_mtr_loss = 0
        total_mtp_loss = 0

        for input_dict in iter(self.trainloader):

            input = input_dict['input_sequence'].float()#.cuda()
            weigths = input_dict['weigths'].float()#.cuda()
            type_token = input_dict['type_token'].long()#.cuda()
            spatial_token = input_dict['spatial_token'].long()#.cuda()
            temporal_token = input_dict['temporal_token'].long()#.cuda()
            meta = input_dict['meta']

            model.zero_grad()

            output = model(input,type_token,spatial_token,temporal_token)
                
            loss = 0
            mpr_loss = 0
            mpp_loss = 0
            mtr_loss = 0
            mtp_loss = 0

            if 'MPR' in self.train_tasks:
                mpr_loss = self.oks_loss(output['MPR_output'],input_dict['MPR_GT'].float().cuda(),input_dict['MPR_weights'].float().cuda())
                loss += mpr_loss

            if 'MPP' in self.train_task:
                mpp_loss = self.oks_loss(output['MPP_output'],input_dict['MPP_GT'].float())#.cuda())
                loss += mpp_loss

            if 'MTR' in self.train_task:
                mtr_loss = self.giou_loss(output['MTR_output'],input_dict['MTR_GT'].float().cuda())
                loss += mtr_loss

            if 'MTP' in self.train_task:
                mtp_loss = self.giou_loss(output['MTP_output'],input_dict['MTP_GT'].float().cuda())
                loss += mtp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss
            total_mpr_loss += mpr_loss
            total_mpp_loss += mpp_loss
            total_mtr_loss += mtr_loss
            total_mtp_loss += mtp_loss
            iteration += 1

            if iteration % 100 == 0:
                self.logger.info("iter {} (epoch {}), total_loss = {:.4f}, mpr_loss = {:.4f},mpp_loss = {:.4f},mtr_loss = {:.4f},mtp_loss = {:.4f}"
                        .format(iteration, epoch, loss.item(),mpr_loss.item(),loss.item(),mtp_loss.item()))

            if iteration % 1000 == 0:
                pass
                # pred_local = visualize_local_tracklets()
                # gt_local = visualize_local_tracklets()
                # pred_global = visualize_global_tracklets()
                # gt_global = visualize_global_tracklets()

        return [total_loss,total_mpr_loss,total_mpp_loss,total_mtr_loss,total_mtp_loss], iteration

    def eval_batch(self,model,epoch,test_tasks):
                        # Set to evaluation mode (randomly sample z from the whole distribution)
        all_rec_err_l1 = []
        all_rec_err_l2 = []
        all_pre_err_l1 = []
        all_pre_err_l2 = []
        all_meta = []

        with torch.no_grad():
            for i,input_dict in enumerate(tqdm(self.testloader)):
                pass
            
    def inference(self):

        pass

    def train_eval(self):

        gpu_ids = range(torch.cuda.device_count())
        self.logger.info('Number of GPUs in use {}'.format(gpu_ids))
        model = AnoPoseFormer(tracklet_len=self.opt.tracklet_len,headless=self.opt.headless,embed_dim=self.opt.embed_dim, depth=self.opt.depth,tasks=self.opt.train_tasks)#.cuda()
        
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

            # if epoch%self.opt.eval_interval==0:
            #     self.logger.info('Start evaluation!')
            #     model.eval()
            #     total_eval_loss = self.eval_batch(model,epoch, self.opt.test_tasks)
            #     self.logger.info("epoch {}, total_val_loss: {:.4f}, avg_val_loss: {:.4f}")

        self.logger.info('End Training!')

if __name__ == '__main__':

    opt = parse_opts()
    print (opt)

    pipeline = Train_Eval_Inference(opt)

    if opt.inference:
        pipeline.inference()
    else:
        pipeline.train_eval()
