
from collections import OrderedDict
from typing import Tuple

from time import time, sleep
import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.data_augmentation.fixmatch_aug_nnunet import instane_3D_augmentation_params, weak_3D_augmentation_params, get_default_augmentationUn

from nnunet.training.loss_functions.deep_supervision_weight import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset

from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.loss_functions.fixmatch_loss import weighted_DC_and_CE_loss

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.dataloading.dataset_loading import load_dataset 
from nnunet.training.dataloading.dataset_loading_fixmatch import DataLoader3DFixmatch
import torch.backends.cudnn as cudnn
from nnunet.utilities.tensor_utilities import sum_tensor

import matplotlib.pyplot as plt
import matplotlib

from nnunet.network_architecture.generic_UNet_dualtask import Generic_UNet as Generic_UNet_dualtask
from torch.optim.lr_scheduler import _LRScheduler
import SimpleITK as sitk

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.nn import MSELoss
from tensorboardX import SummaryWriter
from nnunet.training.dataloading.dataset_loading import DataLoader3D

def compute_sdf(img_gt, out_shape):
    """
    adopt from https://github.com/HiLab-git/DTC/tree/master/code
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]): # channel size
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                
                max_posdis = np.max(posdis)
                min_posdis = np.min(posdis)
                negmask = ~posmask
                if negmask.any():
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)

                    negdis = distance(negmask)

                    max_negdis = np.max(negdis)
                    min_negdis = np.min(negdis)

                    sdf = (negdis-min_negdis)/(max_negdis-min_negdis) - (posdis-min_posdis)/(max_posdis-min_posdis)
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

                    # if abs(min_negdis - max_negdis) < 1e-3:
                    #     print('negdis', min_negdis, max_negdis)
                    # if abs(min_posdis - max_posdis) < 1e-3:
                    #     print('posdis', min_posdis, max_posdis)
                else:
                    sdf = - (posdis-min_posdis)/(max_posdis-min_posdis)
                    normalized_sdf[b][c] = sdf

    return normalized_sdf

def get_batch_sdf(target):
    if isinstance(target, list):
        gt_dis = []
        for target_i in target:
            target_dis = compute_sdf(target_i.cpu().numpy(), target_i.shape)
            target_dis = torch.from_numpy(target_dis).float()
            gt_dis.append(target_dis)

    else:
        gt_dis = compute_sdf(target[:].cpu(
        ).numpy(), target.shape)
        gt_dis = torch.from_numpy(gt_dis).float()  # .cuda()

    return gt_dis

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class nnUNetTrainerV2_SemiSu_DTC(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, labeled_batch_size=2, unlabeled_batch_size=2):
        super(nnUNetTrainerV2_SemiSu_DTC, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
      # self.num_batches_per_epoch = 1
        
        self.neglect_th = 0.6
        self.epoch_upperbound = 800 #2000
        self.epoch_lowerbound = -20
        self.mirror_tta = [(2),(3),(4),(2,3),(2,4),(3,4)]
        self.reversed_mirror_tta = [(2),(3),(4),(3,2),(4,2),(4,3)] 
        
        self.loss = weighted_DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.save_every = 20
        self.model_on_vali = False
        self.max_num_epochs = 1010
        self.all_tr_losses_1 = []
        self.all_val_losses_tr_mode_1 = []
        self.all_val_losses_1 = []
        
        self.loss_sdf = MSELoss()
        # self.loss_sdf = MultipleOutputLoss2(self.loss_sdf, self.ds_loss_weights)
        self.weight_loss_sdf = 0.3
        self.loss_consistency = MSELoss()
        self.loss_consistency = MultipleOutputLoss2(self.loss_consistency, self.ds_loss_weights)
        
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        # consistency default 1.0
        # consistency_rampup default 44.0
        return 1.0 * sigmoid_rampup(epoch, 100) # self.max_num_epochs // 5

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet_dualtask(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def get_unlabeled_generators(self, key):
        if key == 'weak':
            self.load_unlabeled_dataset()

            dl_tr_un = DataLoader3D(self.dataset_un, self.basic_generator_patch_size, self.patch_size, self.unlabeled_batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            
        else:
            dl_tr_un = DataLoader3DFixmatch(self.dataset, self.basic_generator_patch_size, self.patch_size, self.unlabeled_batch_size,
                                 False, pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r',output_folder=self.output_folder)

        return dl_tr_un


    def load_unlabeled_dataset(self):

        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                    "_stage%d" % self.stage)

        self.dataset_un = load_dataset(join(self.folder_with_preprocessed_data, 'unlabeled'))


    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)
            self.batch_size = self.labeled_batch_size

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
        
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                self.dl_tr_un_weak = self.get_unlabeled_generators('weak')
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    unpack_dataset(join(self.folder_with_preprocessed_data,'unlabeled'))
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.weak_data_aug_params = weak_3D_augmentation_params
                
                self.tr_gen_un_weak = get_default_augmentationUn(self.dl_tr_un_weak,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    params=self.weak_data_aug_params, 
                    # deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                    # use_nondetMultiThreadedAugmenter=False
                )
                self.strong_data_aug_params = instane_3D_augmentation_params
                self.strong_data_aug_params['do_cutout'] = True
                self.strong_data_aug_params['cutout_range'] = (25,40,40)
                self.strong_data_aug_params['p_cutout'] = 0.8 #0.8
          
                self.print_to_log_file('Unlabled data size',len(self.dataset_un))
                self.print_to_log_file("TRAINING KEYS:\n %s%s" % (str(self.dataset_tr.keys()),str(self.dataset_un.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True


    # need iterate_num_per_epoch
    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True

        # nnUNetTrainer
        # ret = super().run_training()

        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil
        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

        # super(nnUNetTrainer, self).run_training()
        # network_trainer
        _ = self.tr_gen.next()
        _ = self.val_gen.next()
        _ = self.tr_gen_un_weak.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            # do not use progress bar
            for iter_num_per_epoch in range(self.num_batches_per_epoch):
                l = self.run_iteration(self.tr_gen, True, iter_num_per_epoch=iter_num_per_epoch)
                train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))


        self.network.do_ds = ds
        self.tensorboard_writer.close()
        # return ret

    def get_onehot_encode(self, target):

        def onehot_encode(input):
            shp_y = input.shape # B C Z Y X
            shp_x = (self.batch_size, self.num_classes, shp_y[-3], shp_y[-2], shp_y[-1])
            gt = input
            with torch.no_grad():
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                y_onehot.scatter_(1, gt, 1)

            return y_onehot

        if isinstance(target, list):
            out_onehot = [onehot_encode(_) for _ in target]
        else:
            out_onehot = onehot_encode(target)

        return out_onehot

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, iter_num_per_epoch=None):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        regress_lv = 2
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        data_un_weak_dict = next(self.tr_gen_un_weak)
        data_un_weak = data_un_weak_dict['data']
        data_un_weak = maybe_to_torch(data_un_weak)
        
        gt_dis = get_batch_sdf(self.get_onehot_encode(target[regress_lv]))

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            gt_dis = to_cuda(gt_dis)
            data_un_weak = to_cuda(data_un_weak)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                input_data = torch.cat([data, data_un_weak])
                output_data = self.network(input_data)
                
                regress_output = []
                seg_output = []
                regress_output_un = []
                seg_output_un = []
                for idx in range(len(output_data[0])):
                    regress_output.append(output_data[0][idx][:self.labeled_batch_size])
                    seg_output.append(output_data[1][idx][:self.labeled_batch_size])
                    
                    regress_output_un.append(output_data[0][idx][self.labeled_batch_size:])
                    seg_output_un.append(output_data[1][idx][self.labeled_batch_size:])                    
                
                # regress_output, seg_output = self.network(data)
                # del data
                del input_data
                loss_seg = self.loss(seg_output, target) # seg loss
                loss_regress = self.loss_sdf(regress_output[regress_lv], gt_dis)
                dis_to_mask = [softmax_helper(-1500 *_) for _ in regress_output[regress_lv:]] # why * -1500 torch.sigmoid(-1500*outputs_tanh)
                # consistency_loss = torch.mean((dis_to_mask - seg_output) ** 2)
                loss_consistency = self.loss_consistency(dis_to_mask, seg_output[regress_lv:])
                consistency_weight = self.get_current_consistency_weight(self.epoch)  # self.epoch# why 150

                dis_to_mask_un = [softmax_helper(-1500 *_) for _ in regress_output_un[regress_lv:]]
                loss_consistency_un = self.loss_consistency(dis_to_mask_un, seg_output_un[regress_lv:])

                l = loss_seg + self.weight_loss_sdf * loss_regress + consistency_weight * loss_consistency + consistency_weight * loss_consistency_un

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            regress_output, seg_output = self.network(data)
            del data
            # l = self.loss(seg_output, target)

            loss_seg = self.loss(seg_output, target)  # seg loss
            loss_regress = self.loss_sdf(regress_output[regress_lv], gt_dis)
            dis_to_mask = [softmax_helper(-1500 *_) for _ in regress_output[regress_lv:]]  # why * -1500 torch.sigmoid(-1500*outputs_tanh)
            # consistency_loss = torch.mean((dis_to_mask - seg_output) ** 2)
            loss_consistency = self.loss_consistency(dis_to_mask, seg_output[regress_lv:])
            consistency_weight = self.get_current_consistency_weight(self.epoch)
            l = loss_seg + self.weight_loss_sdf * loss_regress + consistency_weight * loss_consistency

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()


        if iter_num_per_epoch != None:
            iter_num = iter_num_per_epoch + self.num_batches_per_epoch * self.epoch
            # print(iter_num, loss_seg.cpu().data.numpy(), loss_regress.cpu().data.numpy(), loss_consistency.cpu().data.numpy(), consistency_weight)
            # self.tensorboard_writer.add_scalar('lr', self.lr, iter_num)
            self.tensorboard_writer.add_scalar('loss/loss', l, iter_num)
            self.tensorboard_writer.add_scalar('loss/loss_seg', loss_seg, iter_num) # CE and DICE loss
            # self.tensorboard_writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            self.tensorboard_writer.add_scalar('loss/loss_regress', loss_regress, iter_num)
            self.tensorboard_writer.add_scalar('loss/consistency_loss', loss_consistency, iter_num)
            self.tensorboard_writer.add_scalar('loss/consistency_loss_un', loss_consistency_un, iter_num)


        if run_online_evaluation:
            self.run_online_evaluation(seg_output, target)

        del target

        return l.detach().cpu().numpy()

    # def initialize_optimizer_and_scheduler(self):
    #     assert self.network is not None, "self.initialize_network must be called first"
    #     self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
    #                                       amsgrad=True)
    #     self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """ # sum_tensor((output_softmax[:,1] == c).float() * (target == c).float(), axes=axes)
        target = target[0]
        output = output[0]
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

class nnUNetTrainerV2_SemiSu_DTC_2(nnUNetTrainerV2_SemiSu_DTC):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, labeled_batch_size=2, unlabeled_batch_size=2):
        super(nnUNetTrainerV2_SemiSu_DTC_2, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)     
        
        self.loss = weighted_DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        
        self.max_num_epochs = 500
        
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.save_every = 20
        self.model_on_vali = False
        self.all_tr_losses_1 = []
        self.all_val_losses_tr_mode_1 = []
        self.all_val_losses_1 = []
        
        self.loss_sdf = MSELoss()
        # self.loss_sdf = MultipleOutputLoss2(self.loss_sdf, self.ds_loss_weights)
        self.weight_loss_sdf = 1.0
        self.loss_consistency = MSELoss()
        self.loss_consistency = MultipleOutputLoss2(self.loss_consistency, self.ds_loss_weights)
        
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        # consistency default 1.0
        # consistency_rampup default 44.0
        return 1.0 * sigmoid_rampup(epoch, 400) # self.max_num_epochs // 5


class nnUNetTrainerV2_DTC(nnUNetTrainerV2_SemiSu_DTC):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, labeled_batch_size=2, unlabeled_batch_size=2):
        super(nnUNetTrainerV2_DTC, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)     
        
        self.loss = weighted_DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        
        self.max_num_epochs = 500
        
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.save_every = 20
        self.model_on_vali = False
        self.all_tr_losses_1 = []
        self.all_val_losses_tr_mode_1 = []
        self.all_val_losses_1 = []
        
        self.loss_sdf = MSELoss()
        # self.loss_sdf = MultipleOutputLoss2(self.loss_sdf, self.ds_loss_weights)
        self.weight_loss_sdf = 1.0
        self.loss_consistency = MSELoss()
        self.loss_consistency = MultipleOutputLoss2(self.loss_consistency, self.ds_loss_weights)
        
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')
    
    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        # consistency default 1.0
        # consistency_rampup default 44.0
        return 1.0 * sigmoid_rampup(epoch, 400) # self.max_num_epochs // 5
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, iter_num_per_epoch=None):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        regress_lv = 2
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        # data_un_weak_dict = next(self.tr_gen_un_weak)
        # data_un_weak = data_un_weak_dict['data']
        # data_un_weak = maybe_to_torch(data_un_weak)
        
        gt_dis = get_batch_sdf(self.get_onehot_encode(target[regress_lv]))

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            gt_dis = to_cuda(gt_dis)
            # data_un_weak = to_cuda(data_un_weak)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():

                regress_output, seg_output = self.network(data)
                del data
                loss_seg = self.loss(seg_output, target) # seg loss
                loss_regress = self.loss_sdf(regress_output[regress_lv], gt_dis)
                dis_to_mask = [softmax_helper(-1500 *_) for _ in regress_output[regress_lv:]] # why * -1500 torch.sigmoid(-1500*outputs_tanh)
                # consistency_loss = torch.mean((dis_to_mask - seg_output) ** 2)
                loss_consistency = self.loss_consistency(dis_to_mask, seg_output[regress_lv:])
                consistency_weight = self.get_current_consistency_weight(self.epoch)  # self.epoch# why 150

                l = loss_seg + self.weight_loss_sdf * loss_regress + consistency_weight * loss_consistency

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            regress_output, seg_output = self.network(data)
            del data
            # l = self.loss(seg_output, target)

            loss_seg = self.loss(seg_output, target)  # seg loss
            loss_regress = self.loss_sdf(regress_output[regress_lv], gt_dis)
            dis_to_mask = [softmax_helper(-1500 *_) for _ in regress_output[regress_lv:]]  # why * -1500 torch.sigmoid(-1500*outputs_tanh)
            # consistency_loss = torch.mean((dis_to_mask - seg_output) ** 2)
            loss_consistency = self.loss_consistency(dis_to_mask, seg_output[regress_lv:])
            consistency_weight = self.get_current_consistency_weight(self.epoch)
            l = loss_seg + self.weight_loss_sdf * loss_regress + consistency_weight * loss_consistency

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if iter_num_per_epoch != None:
            iter_num = iter_num_per_epoch + self.num_batches_per_epoch * self.epoch
            self.tensorboard_writer.add_scalar('loss/loss', l, iter_num)
            self.tensorboard_writer.add_scalar('loss/loss_seg', loss_seg, iter_num) # CE and DICE loss
            self.tensorboard_writer.add_scalar('loss/loss_regress', loss_regress, iter_num)
            self.tensorboard_writer.add_scalar('loss/consistency_loss', loss_consistency, iter_num)


        if run_online_evaluation:
            self.run_online_evaluation(seg_output, target)

        del target

        return l.detach().cpu().numpy()
    

if __name__ == "__main__":
    print("done")