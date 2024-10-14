import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import nibabel as nib

# tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

def map_GS_to_GGG(gs):
    gs_scores_to_groups = {
        6: 1,
        7: 1,
        "7b": 1,
        8: 2,
        9: 3,
        10: 3
    }

    return gs_scores_to_groups[gs]

## 3D dataset of cropped tumors multichannel (one channel per modality)
class TumorPrivateDataset(Dataset):

    def get_file_names(self, patient_id_str):

        file_extension = '.npy'
        # file_extension = '.nii.gz

        adc_image = None
        t2w_image = None
        for image in os.listdir(os.path.join(self.prostate_data_dir, patient_id_str)):
            if '0000.npy' in image:
                t2w_image = image
            elif '0001.npy' in image:
                adc_image = image
            elif '0002.npy' in image:
                diff_image = image
        if adc_image is None or t2w_image is None:
            raise ValueError('Could not find both adc and t2w images for patient' + patient_id_str)

        return t2w_image, adc_image, diff_image
    
    def build_multimodality_image(self, t2w_array, adc_array, diff_array):
            # Check if the input images have the same shape and are 3D
        if t2w_array.shape != adc_array.shape or t2w_array.ndim != 3 or adc_array.ndim != 3:
            raise ValueError("Both images must have the same shape and be 3D")

        height, width, num_slices = t2w_array.shape

        # Initialize the combined image set with PyTorch format [channels, slices, height, width]
        combined_img_set = np.zeros((3, num_slices, height, width), dtype=t2w_array.dtype)

        for i in range(num_slices):
            # Combine the slices from the two images and the black channel
            combined_img_set[0, i, :, :] = t2w_array[:, :, i]  # First channel from img1's slice
            combined_img_set[1, i, :, :] = adc_array[:, :, i]  # Second channel from img2's slice
            combined_img_set[2, i, :, :] = diff_array[:, :, i]  

        return combined_img_set
    

    def __init__(self, root_dir="../data", means=None, stds=None, transform=None, global_min=None, global_max=None):
        self.root_dir = root_dir

        self.prostate_data_dir = os.path.join(root_dir, "NNUNet_Lesion", "Extracted_Tumor_Regions_Margin_Cropped_Numpy")
        self.patients = [directory for directory in os.listdir(self.prostate_data_dir) 
                        if os.path.isdir(os.path.join(self.prostate_data_dir, directory)) and len(os.listdir(os.path.join(self.prostate_data_dir, directory))) > 0]

        label_dir = os.path.join(root_dir, "ProstateData", "BREST GS.xlsx")
        self.labels = pd.read_excel(label_dir)

        self.transform = transform
        self.means = means
        self.stds = stds
        self.global_min = global_min
        self.global_max = global_max

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        patient_id_str = self.patients[idx]
        patient_id = int(patient_id_str)

        t2w_image_name, adc_image_name, diff_image_name = self.get_file_names(patient_id_str)
        t2w_img_file = os.path.join(self.prostate_data_dir, patient_id_str, t2w_image_name)
        adc_img_file = os.path.join(self.prostate_data_dir, patient_id_str, adc_image_name)
        diff_img_file = os.path.join(self.prostate_data_dir, patient_id_str, diff_image_name)
        

        t2w_image = np.load(t2w_img_file)
        adc_image = np.load(adc_img_file)
        diff_image = np.load(diff_img_file)


        # convert to float array
        t2w_image = t2w_image.astype(np.single)
        adc_image = adc_image.astype(np.single)
        diff_image = diff_image.astype(np.single)


        final_image = self.build_multimodality_image(t2w_image, adc_image, diff_image)
        
        label = self.labels[self.labels["ID"] == patient_id]["surgery GS"].item()
        label = map_GS_to_GGG(label) - 1

        data = {'image': final_image, 'label': label}
        return data


class TransformedDataset(Dataset):

    # wrapper class to apply transformations to datasets

    def __init__(self, base_dataset, transform):
        super(TransformedDataset, self).__init__()
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]

        if self.transform:
            final_image = self.transform(data['image'])
            # keep black channel fully black
            final_image[2] = np.zeros_like(final_image[2])
        else:
            final_image = data['image']

        return {'image': final_image, 'label': data['label']}