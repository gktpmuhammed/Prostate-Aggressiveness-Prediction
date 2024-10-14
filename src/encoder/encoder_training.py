import torch
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
from encoder_decoder_model import Autoencoder
from torchvision import transforms
import torchio as tio
import os
import sys
import numpy as np

# add ProstateCancer src directory to sys.path and import dataset
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from dataset.Dataset import ProstateDataset, TransformedDataset, OneSliceDataset, TranformedMaskedDataset


def train_encoder():

    with wandb.init() as run:

        config = run.config

        # Select device to use
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        modality_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 99.5))
        dataset = OneSliceDataset(root_dir="../../data", modality_transform=modality_transform)

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
            #tio.RandomBiasField(coefficients=(0, 0.1))
        ])

        transformed_dataset = TranformedMaskedDataset(dataset, displacement_transform, non_masked_transform)

        # Data loader
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # Model
        model = Autoencoder()
        model.to(device)
        model.train()

        # Loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        lowest_loss = np.inf

        # Training loop
        for epoch in range(config.epochs):
            epoch_loss = 0
            batches = 0
            for batch_data in data_loader:
                data = batch_data["image"]

                # Delete height of slice (since model is 2D)
                data = torch.squeeze(data, dim=2).float().to(device)

                # Input image is the target as well (for autoencoders)
                inputs = data
                targets = data

                outputs = model(inputs)
                
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1
                
            overall_loss = epoch_loss / batches
            wandb.log({"loss": overall_loss})

            print(f'Epoch [{epoch+1}/{config.epochs}], Loss: {overall_loss:.4f}')

            if epoch_loss < lowest_loss:
                torch.save(model.state_dict(), f"checkpoints/encoder.pth")

        print('Training complete')

if __name__=="__main__":

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sweep", action="store_true", help="Use W&B sweep for hyperparameter tuning")
    args = parser.parse_args()
    """

    wandb.login()

    hparams = {
        'lr': 5e-4,
        'batch_size': 16,
        'weight_decay': 0,
        'epochs': 1000
    }

    wandb.init(project="prostate-cancer-encoder", config=hparams)
    train_encoder()