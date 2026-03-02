from os.path import join
from os import listdir
import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii" , ".nii.gz"])

class DatasetFromFolder2D(data.Dataset):
    def __init__(self, labeled_file_dir, num_classes):
        super(DatasetFromFolder2D, self).__init__()
        self.labeled_filenames = [x for x in os.listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes

    def __getitem__(self, index):
        fname1 = self.labeled_filenames[index // len(self.labeled_filenames)]
        fname2 = self.labeled_filenames[index % len(self.labeled_filenames)]

        # 读取第一个图像及标签
        img1 = sitk.ReadImage(join(self.labeled_file_dir, 'image', fname1))
        img1_arr = sitk.GetArrayFromImage(img1)  # (H, W) 或 (1, H, W)
        if img1_arr.ndim == 3:
            img1_arr = img1_arr[0]  # 处理单通道情况
        img1_slice = img1_arr.astype(np.float32)[np.newaxis, :, :]  # (1, H, W)

        lab1 = sitk.ReadImage(join(self.labeled_file_dir, 'label', fname1))
        lab1_arr = sitk.GetArrayFromImage(lab1)
        if lab1_arr.ndim == 3:
            lab1_arr = lab1_arr[0]
        #lab1_slice = self._map_labels(lab1_arr)
        lab1_slice = lab1_arr
        lab1_slice = self.to_categorical(lab1_slice, self.num_classes).astype(np.float32)

        # 读取第二个图像及标签
        img2 = sitk.ReadImage(join(self.labeled_file_dir, 'image', fname2))
        img2_arr = sitk.GetArrayFromImage(img2)
        if img2_arr.ndim == 3:
            img2_arr = img2_arr[0]
        img2_slice = img2_arr.astype(np.float32)[np.newaxis, :, :]

        lab2 = sitk.ReadImage(join(self.labeled_file_dir, 'label', fname2))
        lab2_arr = sitk.GetArrayFromImage(lab2)
        if lab2_arr.ndim == 3:
            lab2_arr = lab2_arr[0]
        #lab2_slice = self._map_labels(lab2_arr)
        lab2_slice = lab2_arr
        lab2_slice = self.to_categorical(lab2_slice, self.num_classes).astype(np.float32)

        return img1_slice, lab1_slice, img2_slice, lab2_slice, fname1, fname2

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n), dtype=np.float32)
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.labeled_filenames) ** 2