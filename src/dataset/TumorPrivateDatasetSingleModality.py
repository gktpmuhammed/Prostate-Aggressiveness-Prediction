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
class TumorPrivateDatasetSingleModality(Dataset):

    def get_file_names(self, patient_id_str):

        file_extension = '.npy'
        # file_extension = '.nii.gz

        adc_image = None
        t2w_image = None
        for image in os.listdir(os.path.join(self.prostate_data_dir, patient_id_str)):
            if '0000.nii' in image:
                t2w_image = image
            elif '0001.nii' in image:
                adc_image = image
            elif '0002.nii' in image:
                diff_image = image
        if adc_image is None or t2w_image is None:
            raise ValueError('Could not find both adc and t2w images for patient' + patient_id_str)

        return t2w_image, adc_image, diff_image
    
    def __init__(self, root_dir="../data", transform=None):
        self.root_dir = root_dir

        self.prostate_data_dir = os.path.join(root_dir, "NNUNet_Lesion", "Private_Extracted_1_Slice")
        self.patients = [directory for directory in os.listdir(self.prostate_data_dir) 
                        if os.path.isdir(os.path.join(self.prostate_data_dir, directory)) and len(os.listdir(os.path.join(self.prostate_data_dir, directory))) > 0]

        label_dir = os.path.join(root_dir, "ProstateData", "BREST GS.xlsx")
        self.labels = pd.read_excel(label_dir)

        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        patient_id_str = self.patients[idx]
        patient_id = int(patient_id_str)

        t2w_image_name, adc_image_name, diff_image_name = self.get_file_names(patient_id_str)
        # t2w_img_file = os.path.join(self.prostate_data_dir, patient_id_str, t2w_image_name)
        adc_img_file = os.path.join(self.prostate_data_dir, patient_id_str, adc_image_name)
        # diff_img_file = os.path.join(self.prostate_data_dir, patient_id_str, diff_image_name)
        

        # t2w_image = np.load(t2w_img_file)
        img = nib.load(adc_img_file)
        img = img.get_fdata()
        # diff_image = np.load(diff_img_file)


        # convert to float array
        # t2w_image = t2w_image.astype(np.single)
        adc_image = img.astype(np.float32)
        adc_image = np.transpose(adc_image, (2,0,1))
        mean = adc_image.mean()
        std = adc_image.std()
        adc_image = (adc_image - mean) / (std + 1e-6)
        adc_image = np.repeat(adc_image, 3, axis=0)


        label = self.labels[self.labels["ID"] == patient_id]["surgery GS"].item()
        label = map_GS_to_GGG(label) - 1

        data = {'image': adc_image, 'label': label}
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
        else:
            final_image = data['image']

        return {'image': final_image, 'label': data['label']}