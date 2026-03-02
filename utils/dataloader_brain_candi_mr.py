from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])

def imgnorm(N_I, index1=0.05, index2=0.05):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min+1e-6)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


def limit(image):
    max = np.where(image < 0)
    image[max] = 0
    return image

def Nor(data):
    data = np.asarray(data)
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, file_dir, num_classes, shot=-1):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(file_dir, 'image')) if is_image_file(x)]
        self.file_dir = file_dir
        self.num_classes = num_classes
        self.list = [0,2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60]
        if shot==-1:
            self.labeled_filenames = self.labeled_filenames
        else:
            self.labeled_filenames = self.labeled_filenames[:int(len(self.labeled_filenames)*shot)]

    def __getitem__(self, index):
        img = sitk.ReadImage(join(self.file_dir, 'image', self.labeled_filenames[index]))
        img = sitk.GetArrayFromImage(img)
        img = Nor(limit(img))

        lab = sitk.ReadImage(join(self.file_dir, 'label', self.labeled_filenames[index]))
        lab = sitk.GetArrayFromImage(lab)
        mask = np.where(lab > 0, 1, 0).astype(np.float32)
        img = img * mask
        img = img[np.newaxis, :, :, :]

        lab = [np.where(lab == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        lab = np.concatenate(lab, axis=0)
        lab = lab.astype(np.float32)
        img = img.astype(np.float32)

        return img, lab, self.labeled_filenames[index]
    def __len__(self):
        return len(self.labeled_filenames)

