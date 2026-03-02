import os
from os import listdir
from os.path import join
import numpy as np
import SimpleITK as sitk

def Get_Ja_2d(displacement):

    D_y = displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1]  
    D_x = displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1] 

    D_x_y = displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1]  # ∂x/∂Y
    D_y_x = displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1]  # ∂y/∂X

    Ja = (D_x + 1) * (D_y + 1) - D_x_y * D_y_x

    return Ja

def Ja(results_dir, model_name, flow_dir):
    image_filenames = listdir(join(results_dir, model_name, flow_dir))
    ja = np.zeros(len(image_filenames))

    for i in range(len(image_filenames)):
        name = image_filenames[i]
        flow = sitk.ReadImage(join(results_dir, model_name, flow_dir, name))
        flow = sitk.GetArrayFromImage(flow)

        flow = sitk.GetImageFromArray(flow)
        ja1 = sitk.DisplacementFieldJacobianDeterminant(flow)
        ja1 = sitk.GetArrayFromImage(ja1)
        count = np.where(ja1 <= 0, 1, 0)
        ja[i] = np.sum(count)/(np.sum(np.ones_like(count)))

        print(name, ja[i])
    return np.mean(ja), np.std(ja)

def Ja_2d(results_dir, model_name, flow_dir):
    image_filenames = listdir(join(results_dir, model_name, flow_dir))
    ja = np.zeros(len(image_filenames))

    for i in range(len(image_filenames)):
        name = image_filenames[i]
        flow = sitk.ReadImage(join(results_dir, model_name, flow_dir, name))
        flow = sitk.GetArrayFromImage(flow)  # 2D displacement field shape: [H, W, 2] or [2, H, W]

        if flow.ndim == 3:
            flow = flow.transpose(1, 2, 0) 

        flow = sitk.GetImageFromArray(flow, isVector=True) 
        ja1 = sitk.DisplacementFieldJacobianDeterminant(flow)
        ja1 = sitk.GetArrayFromImage(ja1)

        count = np.where(ja1 <= 0, 1, 0)
        ja[i] = np.sum(count) / np.prod(ja1.shape)

        print(name, ja[i])
    return np.mean(ja), np.std(ja)

if __name__ == '__main__':
    meanJa, stdJa = Ja_2d(r'xxx', "xxx", 'xxx')
    print(meanJa, '±' , stdJa)