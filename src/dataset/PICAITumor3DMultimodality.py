import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom

# tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

def map_GS_to_GGG(gs):
    gs_scores_to_groups = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        "7b": 1,
        8: 2,
        9: 3,
        10: 3
    }

    return gs_scores_to_groups[gs]

class PICAI3DMultimodality(Dataset):

    def get_file_names(self, patient_id_str):

        file_extension = '.npy'
        # file_extension = '.nii.gz

        adc_image = None
        t2w_image = None
        mask_image = None
        for image in os.listdir(os.path.join(self.prostate_data_dir, patient_id_str)):
            if '_0000.npy' in image:
                t2w_image = image
            elif '_0001.npy' in image:
                adc_image = image
            elif '_0002.npy' in image:
                diff_image = image
            else:
                mask_image = image
        if adc_image is None or t2w_image is None:
            raise ValueError('Could not find both adc and t2w images for patient' + patient_id_str)

        return t2w_image, adc_image, diff_image, mask_image

        
    def get_label_for_patient(self, patient_id):
        lesion_gs_value_csv = self.labels_csv.loc[self.labels_csv['patient_id'] == patient_id, 'lesion_GS'].values
        gleason_grades = lesion_gs_value_csv[0].split(',')

        max_score = 0 
        for grade in gleason_grades:
            current_score = 0
            for character in grade:
                if character.isdigit():
                    current_score += int(character)
            if current_score > max_score:
                max_score = current_score
        
        return map_GS_to_GGG(max_score) - 1 
    
    def build_multimodality_image(self, t2w_array, adc_array, diff_array, mask_image):
            # Check if the input images have the same shape and are 3D
        if t2w_array.shape != adc_array.shape or t2w_array.ndim != 3 or adc_array.ndim != 3:
            raise ValueError("Both images must have the same shape and be 3D")

        height, width, num_slices = t2w_array.shape

        # Initialize the combined image set with PyTorch format [channels, slices, height, width]
        combined_img_set = np.zeros((4 if self.include_mask else 3, num_slices, height, width), dtype=t2w_array.dtype)

        for i in range(num_slices):
            # Combine the slices from the two images and the black channel
            combined_img_set[0, i, :, :] = t2w_array[:, :, i]  # First channel from img1's slice
            combined_img_set[1, i, :, :] = adc_array[:, :, i]  # Second channel from img2's slice
            combined_img_set[2, i, :, :] = diff_array[:, :, i]

            if self.include_mask:
                combined_img_set[3, i, :, :] = mask_image[:, :, i]  

        if self.modality_transform:
            for i in range(3):  # Iterate over channels (except mask)
                t = np.expand_dims(combined_img_set[i], axis=0)
                if not np.all(t == t.flat[0]):
                    t = self.modality_transform(t)
                t = np.squeeze(t, axis=0)
                combined_img_set[i] = t

        return combined_img_set
    

    def __init__(self, root_dir="../../data", version="NNUNet_Lesion/Picai_Extracted_Regions_AI_Numpy", include_mask=False, modality_transform=None):
        self.root_dir = root_dir

        self.prostate_data_dir = os.path.join(root_dir, version)
        self.patients = [directory for directory in os.listdir(self.prostate_data_dir) 
                        if os.path.isdir(os.path.join(self.prostate_data_dir, directory)) and len(os.listdir(os.path.join(self.prostate_data_dir, directory))) > 0]

        label_dir = os.path.join(root_dir, "PICAIDataset", "input", "picai_labels", "clinical_information", "marksheet.csv")
        self.labels_csv = pd.read_csv(label_dir)

        self.modality_transform = modality_transform

        self.include_mask = include_mask

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        patient_id_str = self.patients[idx]
        patient_id = int(patient_id_str)

        t2w_image_name, adc_image_name, diff_image_name, mask_image_name = self.get_file_names(patient_id_str)
        t2w_img_file = os.path.join(self.prostate_data_dir, patient_id_str, t2w_image_name)
        adc_img_file = os.path.join(self.prostate_data_dir, patient_id_str, adc_image_name)
        diff_img_file = os.path.join(self.prostate_data_dir, patient_id_str, diff_image_name)
        mask_img_file = os.path.join(self.prostate_data_dir, patient_id_str, mask_image_name)
        

        t2w_image = np.load(t2w_img_file)
        adc_image = np.load(adc_img_file)
        diff_image = np.load(diff_img_file)
        mask_image = np.load(mask_img_file)

        # convert to float array
        t2w_image = t2w_image.astype(np.single)
        adc_image = adc_image.astype(np.single)
        diff_image = diff_image.astype(np.single)
        mask_image = mask_image.astype(np.single)

        final_image = self.build_multimodality_image(t2w_image, adc_image, diff_image, mask_image)

        label = self.get_label_for_patient(patient_id)

        data = {'image': final_image, 'label': label}
        return data
    

class PICAITumor3DMultimodality(PICAI3DMultimodality):
    def __init__(self, root_dir="../../data", modality_transform=None, version="NNUNet_Lesion/Picai_Extracted_Regions_AI_Numpy"):
        super().__init__(root_dir=root_dir, version=version, modality_transform=modality_transform)