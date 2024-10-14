from dataset.PICAITumor3DMultimodality import PICAI3DMultimodality
from dataset.Dataset import ProstateDataset, TransformedDataset, TranformedMaskedDataset, OneSliceDataset, TumorOnlyDataset
from network.densenet import DenseNet3D
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from network.cnn import Simple3DCNN, ThreeSliceCNN
from network.rad_image_net import Backbone, Classifier, ThreeSliceBackbone, ThreeSliceClassifier, SliceProcessor, AttentionModel, MultiChannelBackbone, MultiChannelClassifier
from network.medicalnet import MedicalNetClassifier, MedicalNetClassifierMultimodal
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torchvision.models as models
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torchio as tio
import torch
from torch import nn
import wandb
import argparse
import numpy as np
from config.ConfigLoader import load_config
from sklearn.model_selection import train_test_split, StratifiedKFold

import os
# optional: make cuda operations synchronous to get an accurate stack trace in case of cuda errors
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train_cycle(model, train_loader, loss_func, optimizer, device):
    model.train()
    total_loss = 0

    all_outputs = []
    all_labels = []

    for batch in train_loader:
        optimizer.zero_grad()
        data, labels = batch["image"], batch["label"]
        data, labels = data.float().to(device), labels.to(device) # Classification
        # data, labels = data.float().to(device), labels.float().to(device) # Regression
        
        output = model(data) # Classification
        # output = model(data).flatten() # Regression

        loss = loss_func(output, labels)

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        _, pred = torch.max(output, dim=1) # CLASSIFICATION
        # pred = torch.round(output).clamp(min=0, max=4) # REGRESSION
        all_outputs.append(pred)
        all_labels.append(labels)

    all_outputs_tensor = torch.cat(all_outputs, dim=0).clone().detach().cpu()
    all_labels_tensor = torch.cat(all_labels, dim=0).clone().detach().cpu()

    acc = accuracy_score(all_labels_tensor, all_outputs_tensor)
    balanced_acc = balanced_accuracy_score(all_labels_tensor, all_outputs_tensor)

    total_loss /= len(train_loader)
    return total_loss, acc, balanced_acc


def val_cycle(model, val_loader, loss_func, device):
    model.eval()
    val_loss = 0

    all_outputs = []
    all_labels = []

    for batch in val_loader:
        data, labels = batch["image"], batch["label"]
        data, labels = data.float().to(device), labels.to(device) # Classification
        # data, labels = data.float().to(device), labels.float().to(device) # Regression
        
        output = model(data) # Classification
        # output = model(data).flatten() # Regression

        loss = loss_func(output, labels)
        val_loss += loss.item()

        _, pred = torch.max(output, dim=1) # Classification
        # pred = torch.round(output).clamp(min=0, max=4) # REGRESSION
        all_outputs.append(pred)
        all_labels.append(labels)

    all_outputs_tensor = torch.cat(all_outputs, dim=0).clone().detach().cpu()
    all_labels_tensor = torch.cat(all_labels, dim=0).clone().detach().cpu()

    acc = accuracy_score(all_labels_tensor, all_outputs_tensor)
    balanced_acc = balanced_accuracy_score(all_labels_tensor, all_outputs_tensor)

    val_loss /= len(val_loader)
    return val_loss, acc, balanced_acc


def load_radimagenet(num_classes):
    backbone = Backbone()
    classifier = Classifier(num_class=num_classes)
    backbone.load_state_dict(torch.load("../data/RadImageNet/pretrained/RadImageNet_pytorch/ResNet50.pt"))

    model = nn.Sequential(backbone, classifier)
    print("model:", model)
    for param_name, param in model.named_parameters():
        print("param name:", param_name)

    return model


def load_multichannel_radimagenet(num_classes, num_channels):
    backbone = MultiChannelBackbone(num_channels=num_channels, weight_path="../data/RadImageNet/pretrained/RadImageNet_pytorch/ResNet50.pt")
    classifier = MultiChannelClassifier(num_class=num_classes, num_channels=num_channels)

    model = nn.Sequential(backbone, classifier)
    print("model:", model)
    for param_name, param in model.named_parameters():
        print("param name:", param_name)

    return model
    

def load_radimagenet_3slice(num_classes):
    backbone = ThreeSliceBackbone(weight_path="../data/RadImageNet/pretrained/RadImageNet_pytorch/ResNet50.pt")
    classifier = ThreeSliceClassifier(num_class=num_classes)

    model = nn.Sequential(backbone, classifier)
    print("model:", model)
    for param_name, param in model.named_parameters():
        print("param name:", param_name)

    return model


def load_medicalnet_classifier(num_channels, num_classes, device):
    # https://github.com/Tencent/MedicalNet/issues/58

    if num_channels > 1:
        model = MedicalNetClassifierMultimodal(path_to_weights="../data/MedicalNet/pretrain/resnet_34.pth", device=device, num_classes=num_classes, num_channels=num_channels)
    else:
        model = MedicalNetClassifier(path_to_weights="../data/MedicalNet/pretrain/resnet_34.pth", device=device, num_classes=num_classes)

    print("model:", model)
    for param_name, param in model.named_parameters():
        print("param name:", param_name)

    return model

def load_resnet_50(num_classes, device):

    resnet50 = models.resnet50(pretrained=False)  # Set pretrained=False to train from scratch
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, num_classes)

    return resnet50


def get_train_val_indices(cross_val, num_splits, split, labels, train_proportion):
    if cross_val:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=10)
        train_indices = []
        val_indices = []
        for train_index, val_index in skf.split(np.arange(len(labels)), labels):
            train_indices.append(train_index)
            val_indices.append(val_index)

        print("train_indices_before:", train_indices)

        train_indices = train_indices[split]
        val_indices = val_indices[split]

        print("train_indices:", train_indices)
    else:
        train_indices, val_indices = train_test_split(
            np.arange(len(labels)),
            test_size=(1-train_proportion),  # or a float representing the proportion
            stratify=labels,
            random_state=10
        )

    return train_indices, val_indices


def get_augmented_train_dataset(dataset_train_, dataset_name):
    # Define data augmentation transformations
    if dataset_name in ["PRIVATE_1_SLICE", "PICAI_1_SLICE"]:
        # for all channels
        displacement_transform = tio.Compose([
            tio.RandomFlip(axes=(0,)),  # equivalent to horizontal flip; axes can be adjusted for 3D
            tio.RandomAffine(scales=(0.9, 1.1), degrees=(-7, 7, 0, 0, 0, 0)),  # for rotation and scaling
        ])
        # for non-mask channels
        non_masked_transform = tio.Compose([
            tio.RandomGamma(log_gamma=(-0.3,0.3)), # contrast
            tio.RandomBlur(std=(0, 0.05)),  # for Gaussian blurring
            tio.RandomNoise(mean=0, std=(0, 0.05)), # for Gaussian noise
        ])
        dataset_train = TranformedMaskedDataset(dataset_train_, displacement_transform, non_masked_transform)

    elif dataset_name in ["PRIVATE_3_SLICE_TUMOR", "PICAI_3_SLICE_TUMOR", "PICAI_ONLY_TUMOR"]:
        train_transform = tio.Compose([
            tio.RandomFlip(axes=(0,)),  # equivalent to horizontal flip; axes can be adjusted for 3D
            tio.RandomGamma(log_gamma=(-0.3,0.3)), # contrast
            tio.RandomBlur(std=(0, 0.05)),  # for Gaussian blurring
        ])
        dataset_train = TransformedDataset(dataset_train_, train_transform)

    else:
        train_transform = tio.Compose([
            tio.RandomFlip(axes=(0,)),  # equivalent to horizontal flip; axes can be adjusted for 3D
            tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=5),  # for rotation and scaling
            tio.RandomElasticDeformation(max_displacement=3),
            tio.RandomGamma(log_gamma=(-0.3,0.3)), # contrast
            tio.RandomBlur(std=(0, 0.05)),  # for Gaussian blurring
            tio.RandomNoise(mean=0, std=(0, 0.03)), # for Gaussian noise
            tio.RandomBiasField(coefficients=(0, 0.2))
        ])
        dataset_train = TransformedDataset(dataset_train_, train_transform)

    return dataset_train


def get_class_weights(dataset_train, dataset_val, num_classes):

    # get class weights using training set
    train_labels = [d["label"] for d in dataset_train]
    train_labels_counted = [0] * num_classes
    for i in range(0, num_classes):
        count_i = 0
        for j in train_labels:
            if j == i:
                count_i += 1
        train_labels_counted[i] = count_i
    train_class_counts = torch.tensor(train_labels_counted, dtype=torch.float32)
    print('train class counts', train_class_counts)

    # only for debugging
    val_labels = [d["label"] for d in dataset_val]
    val_labels_counted = [0] * num_classes
    for i in range(0, num_classes):
        count_i = 0
        for j in val_labels:
            if j == i:
                count_i += 1
        val_labels_counted[i] = count_i
    val_class_counts = torch.tensor(val_labels_counted, dtype=torch.float32)
    print('val class counts', val_class_counts)

    weights = 1. / train_class_counts
    print("class weights:", weights)
    return weights


def load_optimizer(model, model_name, config):
    if model_name in ["MEDICALNET", "RADIMAGENET_1_SLICE", "RADIMAGENET_3_SLICES"]:
        if model_name == "MEDICALNET":
            pretrained_weights_str = "model."
        else:
            pretrained_weights_str = "0.backbone" # for RadImageNet
        if config.train_pretrained_weight > 0:

            lower_lr_params = []
            standard_lr_params = []

            for param_name, param in model.named_parameters():
                if param_name.startswith(pretrained_weights_str):
                    lower_lr_params.append(param)
                else:
                    standard_lr_params.append(param)

            # Set up the optimizer with two parameter groups
            optimizer = torch.optim.Adam([
                {'params': lower_lr_params, 'lr': (config.train_pretrained_weight * config.lr)},
                {'params': standard_lr_params, 'lr': config.lr}
            ], weight_decay=config.weight_decay)
        else:
            for param_name, param in model.named_parameters():
                if param_name.startswith(pretrained_weights_str):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            
            optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return optimizer


def train_model(config=None, cross_val=False, split=0, num_splits=5, early_stopping_patience=np.inf):

    with wandb.init() as run:

        # load cross validation config
        if not config:
            config = run.config
            dataset_name = config.dataset
            model_name = config.model
            num_classes = config.num_classes
            train_portion = config.train_portion
        else:
            dataset_name = config["dataset"]
            model_name = config["model"]
            num_classes = config["num_classes"]
            train_portion = config["train_portion"]
            config = SimpleNamespace(**config["hparams"])
        print("config:", config)

        # Select device to use
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        # Define data preprocessing transformations
        if model_name=="MEDICALNET":
            # MedicalNet dataset is z-normalized
            preprocessing_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        elif model_name in ["RADIMAGENET_1_SLICE", "RADIMAGENET_3_SLICES"]:
            preprocessing_transform = tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0, 99.5))
        else:
            # Rescale intensity values between 0 and 1; don't consider 0.5% with highest intensities
            preprocessing_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 99.5))

        if dataset_name == "PICAI_3_SLICE_TUMOR":
            prostate_dataset = PICAI3DMultimodality(root_dir='../data', version="NNUNet_Lesion/Picai_AI_Extracted_3_Slice_Numpy", modality_transform=preprocessing_transform)
            num_channels = 3
        elif dataset_name == "PICAI_1_SLICE":
            modalities = "t2w+adc+dwi+mask"
            prostate_dataset = PICAI3DMultimodality(root_dir="../data", version="NNUNet_Lesion/Picai_Extracted_1_Slice_Numpy", include_mask=True, modality_transform=preprocessing_transform)
            num_channels = 4
        elif dataset_name == "PICAI_ONLY_TUMOR":
            prostate_dataset = PICAI3DMultimodality(root_dir="../data", modality_transform=preprocessing_transform)
            num_channels = 3
        elif dataset_name == "PRIVATE_ONLY_TUMOR":
            prostate_dataset = TumorOnlyDataset(root_dir="../data", num_classes=num_classes, modality_transform=preprocessing_transform)
            num_channels = 3
        elif dataset_name == "PRIVATE_1_SLICE":
            prostate_dataset = OneSliceDataset(root_dir="../data", num_classes=num_classes, modality_transform=preprocessing_transform)
            num_channels = 4
        elif dataset_name == "PRIVATE_PROSTATE":
            modalities = 't2w+adc+pet'
            include_pca_segmentations = True
            num_channels = len(modalities.split("+")) + (1 if include_pca_segmentations else 0)
            prostate_dataset = ProstateDataset(modalities=modalities, modality_transform=preprocessing_transform, num_classes=num_classes, include_pca_segmentations=include_pca_segmentations)

        labels = [d["label"] for d in prostate_dataset]
        print('labels', len(labels))

        train_indices, val_indices = get_train_val_indices(cross_val, num_splits, split, labels, train_portion)

        # Create PyTorch subsets
        dataset_train_ = Subset(prostate_dataset, train_indices)
        dataset_val = Subset(prostate_dataset, val_indices)

        dataset_train = get_augmented_train_dataset(dataset_train_, dataset_name)

        print("dataset length:", len(dataset_train))

        if model_name == "RADIMAGENET_1_SLICE":
            model = load_multichannel_radimagenet(num_classes=num_classes, num_channels=num_channels)
        elif model_name == "THREE_SLICE_CNN":
            model = ThreeSliceCNN(input_channels=num_channels, num_classes=num_classes)
        elif model_name == "RADIMAGENET_3_SLICES":
            model = load_radimagenet_3slice(num_classes)
        elif model_name == "MEDICALNET":
            model = load_medicalnet_classifier(num_channels, num_classes, device)
        elif model_name == "DENSENET_3D":
            model = DenseNet3D(num_classes=num_classes)
        elif model_name == "SIMPLE_CNN_3D":
            model = Simple3DCNN(num_channels, num_classes)

        # backbone = Backbone(weight_path='/local_ssd/practical_wise24/prostate_cancer/RadImageNet/pretrained/RadImageNet_pytorch/ResNet50.pt')
        # slice_processor = SliceProcessor(backbone)
        # model = AttentionModel(slice_processor)

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("total number of pytorch parameters:", pytorch_total_params)

        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("total number of trainable parameters:", pytorch_trainable_params)


        print('Using {}...\n'.format(device))
        model = model.to(device)

        weights = get_class_weights(dataset_train, dataset_val, num_classes).to(device)
        loss_func = torch.nn.CrossEntropyLoss(weight=weights) # classification

        """
        class_weights = [len(prostate_dataset)/train_class_counts[i] for i in range(len(train_class_counts))]
        # Assign weight to each sample in the dataset
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        """

        train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=16) # add sampler=sampler when not using a weighted loss function
        val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True, num_workers=16)

        print("Dataset loaded!")

        # early stopping if validation_loss doesn't improve for early_stopping_patience epochs
        epochs_without_improvement = 0

        optimizer = load_optimizer(model, model_name, config)

        lowest_loss = np.inf

        for i in range(config.epochs):

            print("training epoch {}/{}".format(i + 1, config.epochs))

            train_loss, acc, bacc = train_cycle(model, train_loader, loss_func, optimizer, device)
            print("loss:", train_loss)

            with torch.no_grad():
                val_loss, vacc, vbacc = val_cycle(model, val_loader, loss_func, device)
                print("val_loss:", val_loss)

            wandb.log({"train_loss": train_loss, "train_accuracy": acc, "train_balanced_accuracy": bacc, "val_loss": val_loss, "val_accuracy": vacc, "val_balanced_accuracy": vbacc})

            if val_loss < lowest_loss:
                # save model with lowest validation loss
                torch.save(model.state_dict(), "model_checkpoints/densenetoverfit.pth")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered because no improvement was made since", i, "epochs")
                break

        # return evaluation metrics at the end of training
        return acc, bacc, vacc, vbacc

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sweep", action="store_true", help="Use W&B sweep for hyperparameter tuning")
    parser.add_argument("--cross_val_splits", type=int, default=1, help="Number of splits for cross validation. Set to 1 if you don't want to use cross-validation.")
    parser.add_argument("--cross_val_output_path", type=str, default="./cross_val_output.txt")
    args = parser.parse_args()

    wandb.login()

    if args.use_sweep:

        print("Preparing hyperparameter search...")

        config = load_config()

        sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'   
            },
            'parameters': {
                'lr': {
                    'min': 1e-7,
                    'max': 2e-2
                },
                'batch_size': {
                    'values': [8]
                },
                'weight_decay': {
                    'min': 0.0,
                    'max': 1e-3
                },
                'epochs': {
                    'values': [200]
                },
                'train_pretrained_weight': {
                    'values': [0.0, 1, 0.1, 0.01, 0.001]
                },
                'model': {
                    'values': [config["model"]]
                },
                'dataset': {
                    'values':  [config["dataset"]]
                },
                'num_classes': {
                    'values':  [config["num_classes"]]
                },
                'train_portion': {
                    'values':  [config["train_portion"]]
                }
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_config, project="prostate-cancer-classification")

        wandb.agent(sweep_id, train_model, count=100)

    elif args.cross_val_splits > 1:

        print("Cross-validating model...")

        config = load_config()

        train_accuracies = []
        train_balanced_accuracies = []
        val_accuracies = []
        val_balanced_accuracies = []

        for i in range(args.cross_val_splits):
            wandb.init(project="prostate-cancer-classification", group="cross-val", tags="cross-validation-run-1", config=config)
            acc, bacc, vacc, vbacc = train_model(config=wandb.config, cross_val=True, split=i, num_splits=args.cross_val_splits)

            train_accuracies.append(acc)
            train_balanced_accuracies.append(bacc)
            val_accuracies.append(vacc)
            val_balanced_accuracies.append(vbacc)

        with open(args.cross_val_output_path, 'w') as file:
            file.write("====== Cross Validation Results ======" + "\n")
            file.write("train_accuracies:" + str(train_accuracies) + "\n")
            file.write("train_balanced_accuracies:" + str(train_balanced_accuracies) + "\n")
            file.write("val_accuracies:" + str(val_accuracies) + "\n")
            file.write("val_balanced_accuracies:" + str(val_balanced_accuracies) + "\n")
            file.write("======================================" + "\n")

    else:

        print("Preparing training...")

        config = load_config()

        wandb.init(project="prostate-cancer-classification", config=config)
        train_model(config=wandb.config)
        