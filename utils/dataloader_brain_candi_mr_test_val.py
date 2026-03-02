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
    def __init__(self, file_dir, num_classes, shape=(160, 160, 128)):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(file_dir, 'image')) if is_image_file(x)]
        self.file_dir = file_dir
        self.num_classes = num_classes
        self.list = [0,2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60]
        self.shape = shape

    def __getitem__(self, index):
        n = len(self.labeled_filenames)

        idx1 = index // n
        idx2 = index % n

        img1 = sitk.ReadImage(join(self.file_dir, 'image', self.labeled_filenames[idx1]))
        img1 = sitk.GetArrayFromImage(img1)
        img1 = Nor(limit(img1))
        lab1 = sitk.ReadImage(join(self.file_dir, 'label', self.labeled_filenames[idx1]))
        lab1 = sitk.GetArrayFromImage(lab1)
        mask1 = np.where(lab1 > 0, 1, 0).astype(np.float32)
        img1 = img1 * mask1
        img1 = img1[np.newaxis, :, :, :]
        lab1 = [np.where(lab1 == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        lab1 = np.concatenate(lab1, axis=0)
        lab1 = lab1.astype(np.float32)
        img1 = img1.astype(np.float32)

        img2 = sitk.ReadImage(join(self.file_dir, 'image', self.labeled_filenames[idx2]))
        img2 = sitk.GetArrayFromImage(img2)
        img2 = Nor(limit(img2))
        lab2 = sitk.ReadImage(join(self.file_dir, 'label', self.labeled_filenames[idx2]))
        lab2 = sitk.GetArrayFromImage(lab2)
        mask2 = np.where(lab2 > 0, 1, 0).astype(np.float32)
        img2 = img2 * mask2
        img2 = img2[np.newaxis, :, :, :]
        lab2 = [np.where(lab2 == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        lab2 = np.concatenate(lab2, axis=0)
        lab2 = lab2.astype(np.float32)
        img2 = img2.astype(np.float32)

        return img1, lab1, img2, lab2, self.labeled_filenames[idx1], self.labeled_filenames[idx2]
    def reshape_img(self, image, label, shape):
        if image.shape[1] <= shape[0]:
            image = np.concatenate([image, np.zeros((image.shape[0], shape[0]-image.shape[1], image.shape[2], image.shape[3]))], axis=1)
            label = np.concatenate([label, np.zeros((label.shape[0], shape[0]-label.shape[1], label.shape[2], label.shape[3]))], axis=1)
            x_idx = 0
        else:
            x_idx = np.random.randint(image.shape[1] - shape[0])

        if image.shape[2] <= shape[1]:
            image = np.concatenate([image, np.zeros((image.shape[0], image.shape[1], shape[1]-image.shape[2], image.shape[3]))], axis=2)
            label = np.concatenate([label, np.zeros((label.shape[0], label.shape[1], shape[1] - label.shape[2], label.shape[3]))], axis=2)
            y_idx = 0
        else:
            y_idx = np.random.randint(image.shape[2] - shape[1])

        if image.shape[3] <= shape[2]:
            image = np.concatenate([image, np.zeros((image.shape[0], image.shape[1], image.shape[2], shape[2]-image.shape[3]))], axis=3)
            label = np.concatenate([label, np.zeros((label.shape[0], label.shape[1], label.shape[2], shape[2] - label.shape[3]))], axis=3)
            z_idx = 0
        else:
            z_idx = np.random.randint(image.shape[3] - shape[2])

        image = image[:, x_idx:x_idx+shape[0], y_idx:y_idx+shape[1], z_idx:z_idx+shape[2]]
        label = label[:, x_idx:x_idx+shape[0], y_idx:y_idx+shape[1], z_idx:z_idx+shape[2]]
        return image, label
    def __len__(self):
         return len(self.labeled_filenames) * len(self.labeled_filenames)

