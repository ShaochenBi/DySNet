import os
from os.path import join
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from model.Ja_nii import Ja_2d
from model.DySNet_X import DySNet_X
from utils.STN import SpatialTransformer
from utils.dataloader_train_2d import DatasetFromFolder2D
from utils.dataloader_test_2d import DatasetFromFolder2D as DatasetFromFolder2D_test_reg
from utils.losses_2d import gradient_loss, ncc_loss, dice_loss, MAE, partical_MAE
from utils.utils import AverageMeter, to_categorical, dice
import numpy as np
import SimpleITK as sitk

class Trainer2D(object):
    def __init__(self, k=0,
                 n_channels=1,
                 lr=1e-4,
                 epoches=200,
                 iters=200,
                 batch_size=1,
                 num_classes=5,
                 checkpoint_dir='/home/sbiab/data/train_2d/DySNet_X',
                 result_dir='/home/sbiab/data/train_2d',
                 train_data_dir='/home/sbiab/data/train_2d',
                 test_data_dir='/home/sbiab/data/test_2d',
                 model_name='DySNet_X',
                 ):
        super(Trainer2D, self).__init__()
        self.k = k
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        self.test_data_dir = test_data_dir
        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.Network = DySNet_X(in_chans=n_channels, out_chans=2, mode="cross")

        if torch.cuda.is_available():
            self.Network = self.Network.cuda()
        self.opt = torch.optim.AdamW(self.Network.parameters(), lr=lr)

        self.stn = SpatialTransformer()

        train_dataset = DatasetFromFolder2D(train_data_dir)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset_reg = DatasetFromFolder2D_test_reg(test_data_dir, self.num_classes)
        self.dataloader_test_reg = DataLoader(test_dataset_reg, batch_size=1)
        self.L_sim = ncc_loss
        self.L_smooth = gradient_loss

        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')

    def train_iterator(self, imgA, imgB):

        flow_A2B, flow_B2A = self.Network(imgA, imgB)
        loss_sim = 0
        loss_smooth = 0

        imgA2B = self.stn(imgA, flow_A2B)
        imgB2A = self.stn(imgB, flow_B2A)

        loss_sim = self.L_sim(imgA2B, imgB) + self.L_sim(imgB2A, imgA)
        loss_smooth = self.L_smooth(flow_A2B) + self.L_smooth(flow_B2A)

        loss_total = loss_sim + loss_smooth

        loss_total.backward()
        self.opt.step()
        self.Network.zero_grad()
        self.opt.zero_grad()
        self.L_smooth_log.update(loss_smooth.data, imgA.size(0))
        self.L_sim_log.update(loss_sim.data, imgA.size(0))

    def train_epoch(self, epoch):
        self.Network.train()
        for i in range(self.iters):
            imgA, imgB = next(self.dataloader_train.__iter__())
            if torch.cuda.is_available():
                imgA = imgA.cuda()
                imgB = imgB.cuda()

            self.train_iterator(imgA, imgB)

            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_sim_log.__str__()])
            print(res)

    def test(self):
        self.Network.eval()
        for i, (mi, ml, fi, fl, name1, name2) in enumerate(self.dataloader_test_reg):
            name1 = name1[0]
            name2 = name2[0]
            if name1 != name2:
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    fi = fi.cuda()
                    ml = ml.cuda()
                    fl = fl.cuda()
                
                    flowA2B, flow_B2A = self.Network(mi, fi)
                    w_m_to_f = self.stn(mi, flowA2B)
                    w_label_m_to_f = self.stn(ml, flowA2B, mode='nearest')

                    flowA2B = flowA2B.data.cpu().numpy()[0]
                    w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                    w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)

                    flowA2B = flowA2B.astype(np.float32)
                    w_m_to_f = w_m_to_f.astype(np.float32)
                    w_label_m_to_f = w_label_m_to_f.astype(np.int8)

                for subfolder in ['flow', 'w_m_to_f', 'w_label_m_to_f']:
                    save_path = os.path.join(self.results_dir, self.model_name, subfolder)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                w_m_to_f_img = sitk.GetImageFromArray(w_m_to_f)
                sitk.WriteImage(w_m_to_f_img, os.path.join(self.results_dir, self.model_name, 'w_m_to_f', f'{name1[:-7]}_to_{name2[:-7]}.nii'))
                w_label_m_to_f_img = sitk.GetImageFromArray(w_label_m_to_f)
                sitk.WriteImage(w_label_m_to_f_img, os.path.join(self.results_dir, self.model_name, 'w_label_m_to_f', f'{name1[:-7]}_to_{name2[:-7]}.nii'))
                flowA2B_img = sitk.GetImageFromArray(flowA2B)
                sitk.WriteImage(flowA2B_img, os.path.join(self.results_dir, self.model_name, 'flow', f'{name1[:-7]}_to_{name2[:-7]}.nii'))

                print(f'Saved: {name1[:-7]}_to_{name2[:-7]}.nii')

    def evaluate_reg(self, n_classes=5):
        DSC_R = np.zeros((n_classes, len(self.dataloader_test_reg)))
        image_filenames = os.listdir(os.path.join(self.results_dir, self.model_name, 'w_label_m_to_f'))
        for i, name in enumerate(image_filenames):

            w_label_path = os.path.join(self.results_dir, self.model_name, 'w_label_m_to_f', name)
            w_label_m_to_f = sitk.ReadImage(w_label_path)
            w_label_m_to_f = sitk.GetArrayFromImage(w_label_m_to_f)
            w_label_m_to_f = to_categorical(w_label_m_to_f, n_classes)

            base_name = os.path.splitext(name)[0]

            if '_to_' not in base_name:
                raise ValueError(f"Filename format unexpected (missing '_to_'): {name}")
            fixed_name = base_name.split('_to_')[1]

            label_path = os.path.join(self.test_data_dir, 'label', fixed_name + '.nii.gz')
            print(f"Reading label file from: {label_path}")

            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")

            fl = sitk.ReadImage(label_path)
            fl = sitk.GetArrayFromImage(fl)
            fl = to_categorical(fl, n_classes)

            for c in range(n_classes):
                DSC_R[c, i] = dice(w_label_m_to_f[c], fl[c])

            print(f"{name}: Dice scores (excluding background): {DSC_R[1:, i]}")

        mean_dice = np.mean(DSC_R[1:, :])
        std_dice = np.std(np.mean(DSC_R[1:, :], axis=0))
        return mean_dice, std_dice
    

    def checkpoint(self, epoch):
        torch.save(self.Network.state_dict(),
                   os.path.join(self.checkpoint_dir, f'{self.model_name}_epoch_{epoch + self.k}.pth'))

    def load(self):
        self.Network.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, f'{self.model_name}_epoch_{self.k}.pth')))

    def train(self):
        for epoch in range(self.epoches - self.k):
            self.L_smooth_log.reset()
            self.L_sim_log.reset()

            self.train_epoch(epoch + self.k)
            if epoch % 20 == 0:
                self.checkpoint(epoch)
        self.checkpoint(self.epoches - self.k)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainer = Trainer2D()
    # trainer.load()
    trainer.train()
    # trainer.test()
    # meanJa, stdJa = Ja_2d(trainer.results_dir, trainer.model_name, 'flow')
    # DSCmean, DSCstd = trainer.evaluate_reg()
    # print(DSCmean, DSCstd)
    # print(meanJa, stdJa)
    