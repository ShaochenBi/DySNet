from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import torch
import numpy as np
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii" , ".nii.gz"])


class DatasetFromFolder2D(data.Dataset):
    def __init__(self, unlabeled_file_dir):
        super(DatasetFromFolder2D, self).__init__()
        self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir

    def __getitem__(self, index):
        img_dir = os.path.join(self.unlabeled_file_dir, 'image')

        random_index1 = np.random.randint(0, len(self.unlabeled_filenames))
        img1_path = os.path.join(img_dir, self.unlabeled_filenames[random_index1])
        img1 = sitk.ReadImage(img1_path)
        img1_array = sitk.GetArrayFromImage(img1)

        if img1_array.ndim == 3:
            slice_idx1 = np.random.randint(0, img1_array.shape[0])
            img1_slice = img1_array[slice_idx1, :, :]
        elif img1_array.ndim == 2:
            img1_slice = img1_array
        else:
            raise ValueError(f"Unsupported image dimension: {img1_array.ndim}")

        #img1_slice = np.clip(img1_slice, 0., 2048.) / 2048.
        img1_slice = img1_slice.astype(np.float32)[np.newaxis, :, :]

        random_index2 = np.random.randint(0, len(self.unlabeled_filenames))
        img2_path = os.path.join(img_dir, self.unlabeled_filenames[random_index2])
        img2 = sitk.ReadImage(img2_path)
        img2_array = sitk.GetArrayFromImage(img2)

        if img2_array.ndim == 3:
            slice_idx2 = np.random.randint(0, img2_array.shape[0])
            img2_slice = img2_array[slice_idx2, :, :]
        elif img2_array.ndim == 2:
            img2_slice = img2_array
        else:
            raise ValueError(f"Unsupported image dimension: {img2_array.ndim}")

        #img2_slice = np.clip(img2_slice, 0., 2048.) / 2048.
        img2_slice = img2_slice.astype(np.float32)[np.newaxis, :, :]
        #sitk.WriteImage(sitk.GetImageFromArray(img2_slice[0]), 'test.nii')
        return img1_slice, img2_slice

    def __len__(self):
        return len(self.unlabeled_filenames)