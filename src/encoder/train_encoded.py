import torch
import wandb
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torchvision import transforms
import torchio as tio
import os
import sys
import numpy as np
from encoder_decoder_model import Autoencoder
from sklearn.model_selection import train_test_split

# add ProstateCancer src directory to sys.path and import dataset
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from dataset.Dataset import ProstateDataset, TransformedDataset, OneSliceDataset, TranformedMaskedDataset
from network.encoder_cnn import Encoded2DCNN


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


def train_encoder():

    with wandb.init() as run:

        config = run.config

        # Select device to use
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        modality_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 99.5))
        num_classes = 2
        dataset = OneSliceDataset(root_dir="../../data", modality_transform=modality_transform, num_classes=num_classes)

        # train test stratified split
        labels = [d["label"] for d in dataset]
        train_indices, val_indices = train_test_split(
                np.arange(len(labels)),
                test_size=0.4,
                stratify=labels,
                random_state=10
            )
        dataset_train_ = Subset(dataset, train_indices)
        dataset_val = Subset(dataset, val_indices)

        # data augmentation
        # for all channels
        displacement_transform = tio.Compose([
            tio.RandomFlip(axes=(0,)),  # equivalent to horizontal flip; axes can be adjusted for 3D
            tio.RandomAffine(scales=(0.9, 1.1), degrees=(-7, 7, 0, 0, 0, 0)),  # for rotation and scaling
        ])
        # for non-mask channels
        non_masked_transform = tio.Compose([
            tio.RandomGamma(log_gamma=(-0.3,0.3)), # contrast
            tio.RandomBlur(std=(0, 0.05)),  # for Gaussian blurring
            tio.RandomNoise(mean=0, std=(0, 0.03)), # for Gaussian noise
        ])
        dataset_train = TranformedMaskedDataset(dataset_train_, displacement_transform, non_masked_transform)

        # Data loader
        train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True, num_workers=16)

        # Autoencoder
        autoencoder = Autoencoder()
        autoencoder.load_state_dict(torch.load("../encoder/checkpoints/encoder.pth"))
        autoencoder.to(device)
        autoencoder.eval()

        # Model
        model = Encoded2DCNN(num_classes=num_classes)
        model.to(device)

        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=get_class_weights(dataset_train, dataset_val, num_classes).to(device))
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        lowest_loss = np.inf

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            batches = 0
            for batch_data in train_loader:
                data = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                # Delete height of slice (since model is 2D)
                data = torch.squeeze(data, dim=2).float()

                # Input image is the encoded image
                inputs = autoencoder.encoder(data)

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1
                
            train_loss = epoch_loss / batches
            wandb.log({"loss_train": train_loss})

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_batches = 0
                for batch_data in val_loader:
                    data = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)
                    data = torch.squeeze(data, dim=2).float()
                    inputs = autoencoder.encoder(data)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_batches += 1

                val_loss /= val_batches
                wandb.log({"loss_val": val_loss})

            # Save the model with the lowest loss
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                torch.save(model.state_dict(), f"checkpoints/cnn_encoded.pth")

            print(f'Epoch [{epoch+1}/{config.epochs}], Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

        print('Training complete')

if __name__=="__main__":

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sweep", action="store_true", help="Use W&B sweep for hyperparameter tuning")
    args = parser.parse_args()
    """

    wandb.login()

    hparams = {
        'lr': 1e-4,
        'batch_size': 16,
        'weight_decay': 1e-2,
        'epochs': 100
    }

    wandb.init(project="prostate-cancer-train-with-encoder", config=hparams)
    train_encoder()