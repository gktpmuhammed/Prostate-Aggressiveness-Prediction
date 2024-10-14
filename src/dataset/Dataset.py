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
        7: 2,
        "7b": 3,
        8: 4,
        9: 5,
        10: 5
    }

    return gs_scores_to_groups[gs]

def map_GS_to_Three_GGG(gs):
    gs_scores_to_groups = {
        6: 0,
        7: 0,
        "7b": 0,
        8: 1,
        9: 2,
        10: 2
    }

    return gs_scores_to_groups[gs]

def binary_GS(gs):
    gs_scores_to_groups = {
        6: 0,
        7: 0,
        "7b": 0,
        8: 1,
        9: 1,
        10: 1
    }

    return gs_scores_to_groups[gs]

class ProstateDataset(Dataset):

    def get_file_name(self, modality, patient_id_str):

        if modality == 't2w':
            return 'prostate_' + patient_id_str + '_T2W' + self.file_extension
        elif modality == 'dwi':
            return 'prostate_' + patient_id_str + '_DWI' + self.file_extension
        elif modality == 'adc':
            return 'prostate_' + patient_id_str + '_ADC' + self.file_extension
        elif modality == 'pet':
            return 'prostate_' + patient_id_str + '_PET' + self.file_extension
        elif modality == 't1w':
            return 'prostate_' + patient_id_str + '_T1W' + self.file_extension
        else:
            raise ValueError("Invalid modality for ProstateCancer dataset:", modality)

    def any_modality_exists(self, directory, patient_id_str):
        # test if for the patient at least one modality exists

        for filename in os.listdir(directory):
            for m in self.modalities:
                if filename == self.get_file_name(m, patient_id_str):
                    return True

        return False

    def get_shape(self, directory, patient_id_str):
        for m in self.modalities:
            img_file = os.path.join(directory, self.get_file_name(m, patient_id_str))

            if os.path.exists(img_file):

                if self.file_extension == '.nii.gz':
                    # for nifti image
                    volume = nib.load(img_file)
                    volume = volume.get_fdata()
                else:
                    #for numpy images
                    volume = np.load(img_file)

                return volume.shape

        raise RuntimeError("No patient data found in directory", directory)

    def get_patient_id_from_directory_name(self, directory):
        return directory.split('_')[1]

    def __init__(self, modalities='t2w', root_dir="../data", dataset_version="NNUNetModel/Clean_Dataset/Extracted_Tumor_Regions_Margin_Cropped_Numpy", pca_version="NNUNet_Lesion/Extracted_Tumor_Regions_Margin_Cropped_Numpy", file_extension=".npy", num_classes = 5, include_pca_segmentations=False, transform=None, modality_transform=None):
        
        if file_extension not in ['.npy', '.nii.gz']:
            raise ValueError('Invalid file extension!')

        self.file_extension = file_extension

        self.modalities = modalities.split("+")

        self.prostate_data_dir = os.path.join(root_dir, dataset_version)

        self.include_pca_segmentations = include_pca_segmentations
        self.pca_segmentation_data_dir = os.path.join(root_dir, pca_version)
        
        self.patients = [directory for directory in os.listdir(self.prostate_data_dir)
                            if self.any_modality_exists(os.path.join(self.prostate_data_dir, directory), self.get_patient_id_from_directory_name(directory))]

        if len(self.patients) == 0:
            raise RuntimeError("No patients found in dataset with given modalities")

        self.data_shape = self.get_shape(os.path.join(self.prostate_data_dir, self.patients[0]), self.get_patient_id_from_directory_name(self.patients[0]))

        self.label_dir = os.path.join(root_dir, "ProstateData", "BREST GS.xlsx")
        self.labels = pd.read_excel(self.label_dir)

        if num_classes not in [2,3,5]:
            raise ValueError("Invalid number of classes!")

        self.num_classes = num_classes
        self.transform = transform
        self.modality_transform = modality_transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        patient = self.patients[idx]
        
        patient_id_str = self.get_patient_id_from_directory_name(patient)
        patient_id = int(patient_id_str)

        channels = len(self.modalities) + (1 if self.include_pca_segmentations else 0)
        image = np.zeros((channels, *self.data_shape))

        for i, m in enumerate(self.modalities):
            img_file = os.path.join(self.prostate_data_dir, patient, self.get_file_name(m, patient_id_str))

            if os.path.exists(img_file):

                if self.file_extension == '.nii.gz':
                    # for nifti image
                    volume = nib.load(img_file)
                    volume = volume.get_fdata()
                else:
                    #for numpy images
                    volume = np.load(img_file)

                # convert to float array
                volume = volume.astype(np.single)

                volume = np.expand_dims(volume, axis=0) # add a channel at dimension 0 (necessary for PyTorch transformations)

                # only apply modality transform if not all pixels have the same value
                if self.modality_transform and not np.all(volume == volume.flat[0]):
                    volume = self.modality_transform(volume)

                image[i] = volume[0]

        if self.include_pca_segmentations:
            mask_file = os.path.join(self.pca_segmentation_data_dir, patient_id_str, patient_id_str + "_" + patient_id_str + self.file_extension)
            if os.path.exists(mask_file):

                if self.file_extension == '.nii.gz':
                    # for nifti image
                    mask_volume = nib.load(mask_file)
                    mask_volume = mask_volume.get_fdata()
                else:
                    #for numpy images
                    mask_volume = np.load(mask_file)

                # convert to float array
                mask_volume = mask_volume.astype(np.single)

                mask_volume = np.expand_dims(mask_volume, axis=0)

                image[len(self.modalities)] = mask_volume

        # correct dimensions from (channels, width, height, depth) to (channels, depth, height, width)
        image = np.transpose(image, (0, 3, 2, 1))

        label = self.labels[self.labels["ID"] == patient_id]["surgery GS"].item()
        if self.num_classes == 5:
            # label is GGG-1
            label = map_GS_to_GGG(label) - 1
        elif self.num_classes == 3:
            label = map_GS_to_Three_GGG(label)
        elif self.num_classes == 2:
            # only two classes
            label = binary_GS(label)

        if self.transform:
            image = self.transform(image)

        data = {'image': image, 'label': label}
        return data

class OneSliceDataset(ProstateDataset):
    
    def __init__(self, root_dir="../data", num_classes=5, transform=None, modality_transform=None):
        super().__init__(modalities='t2w+adc+pet', root_dir=root_dir, dataset_version="NNUNet_Lesion/Private_Extracted_1_Slice", pca_version="NNUNet_Lesion/Private_Extracted_1_Slice", file_extension=".nii.gz", num_classes=num_classes, include_pca_segmentations=True, transform=transform, modality_transform=modality_transform)

    def get_file_name(self, modality, patient_id_str):

        if modality == 't2w':
            return patient_id_str + '_' + patient_id_str + '_0000' + self.file_extension
        elif modality == 'adc':
            return patient_id_str + '_' + patient_id_str + '_0001' + self.file_extension
        elif modality == 'pet':
            return patient_id_str + '_' + patient_id_str + '_0002' + self.file_extension
        else:
            raise ValueError("Invalid modality for OneSliceDataset:", modality)

    def get_patient_id_from_directory_name(self, directory):
        return directory

class TumorOnlyDataset(ProstateDataset):

    def __init__(self, root_dir="../data", num_classes=5, transform=None, modality_transform=None):
        super().__init__(modalities='t2w+adc+pet', root_dir=root_dir, dataset_version="NNUNet_Lesion/Extracted_Tumor_Regions_Margin_Cropped_Numpy", pca_version="NNUNet_Lesion/Extracted_Tumor_Regions_Margin_Cropped_Numpy", file_extension=".npy", num_classes=num_classes, include_pca_segmentations=False, transform=transform, modality_transform=modality_transform)

    def get_file_name(self, modality, patient_id_str):

        if modality == 't2w':
            return patient_id_str + '_' + patient_id_str + '_0000' + self.file_extension
        elif modality == 'adc':
            return patient_id_str + '_' + patient_id_str + '_0001' + self.file_extension
        elif modality == 'pet':
            return patient_id_str + '_' + patient_id_str + '_0002' + self.file_extension
        else:
            raise ValueError("Invalid modality for OneSliceDataset:", modality)

    def get_patient_id_from_directory_name(self, directory):
        return directory
        
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
        return {'image': self.transform(data['image']), 'label': data['label']}

class TranformedMaskedDataset(Dataset):

    # wrapper class to apply transformations not to mask

    def __init__(self, base_dataset, transform_all, transform_non_masked):
        super(TranformedMaskedDataset, self).__init__()
        self.base = base_dataset
        self.transform_all = transform_all
        self.transform_non_masked = transform_non_masked

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        image = data['image']
        if self.transform_all:
            image = self.transform_all(image)

        # apply per-channel transformations
        transformed_channels = []
        for i in range(image.shape[0] - 1):
            channel = image[i:i+1, :, :, :]
            mask = (channel == channel.min())
            if self.transform_non_masked:
                channel = self.transform_non_masked(channel)
            channel[mask] = channel.min()
            transformed_channels.append(channel)

        # don't transform last channel
        transformed_channels.append(image[image.shape[0] - 1 : image.shape[0], :, :, :])

        image = np.concatenate(transformed_channels, axis=0)

        return {'image': image, 'label': data['label']}