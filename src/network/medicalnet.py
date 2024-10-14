import torch
from torch import nn

import sys
sys.path.append("../data")
sys.path.append("../data/MedicalNet")
from MedicalNet import setting as MNSetting
from MedicalNet import model as MNModel
from MedicalNet.models.resnet import resnet34
#import MedicalNet.models


class MedicalNetClassifier(nn.Module):

  def __init__(self, path_to_weights, device, num_classes):
    super(MedicalNetClassifier, self).__init__()
    
    self.model = resnet34(sample_input_D=30, sample_input_H=92, sample_input_W=99, num_seg_classes=num_classes)

    self.model.conv_seg = nn.Sequential(
        nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
        nn.Flatten(start_dim=1),
        nn.Dropout(0.1)
    )
    net_dict = self.model.state_dict()
    pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
    pretrain_dict = {
        k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
      }
    net_dict.update(pretrain_dict)
    self.model.load_state_dict(net_dict)
    self.fc1 = nn.Linear(512, 256)
    self.fc2 = nn.Linear(256, num_classes)

  def forward(self, x):
    features = self.model(x)
    x = torch.relu(self.fc1(features))
    x = self.fc2(x)
    return x

  
class MedicalNetClassifierMultimodal(nn.Module):
  def __init__(self, path_to_weights, device, num_classes, num_channels):
    super(MedicalNetClassifierMultimodal, self).__init__()

    # Initialize a list to hold a model for each channel
    self.models = nn.ModuleList([
      resnet34(sample_input_D=30, sample_input_H=99, sample_input_W=92, num_seg_classes=num_classes)
      for _ in range(num_channels)
    ])

    # Load pretrained weights for each model
    for model in self.models:
      net_dict = model.state_dict()
      pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
      pretrain_dict = {
          k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
      }
      net_dict.update(pretrain_dict)
      model.load_state_dict(net_dict)

      # Remove conv_seg from each model
      model.conv_seg = nn.Sequential()

    # 1x1 Convolution to combine features from each channel
    #self.combine_conv = nn.Conv3d(num_channels * 512, 1, kernel_size=1)

    self.combine_pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

    # Non-linearity (e.g., ReLU)
    self.relu = nn.ReLU()

    # Fully connected layers
    self.fc1 = nn.Linear(num_channels * 512, 256)
    self.fc2 = nn.Linear(256, num_classes)

  def forward(self, x):
    # Process each channel separately
    features_list = []
    for i, model in enumerate(self.models):
        input_i = x[:, i, :, :, :].unsqueeze(1)
        channel_feature = model(input_i)
        features_list.append(channel_feature)

    # Concatenate the features from each channel
    combined_features = torch.cat(features_list, dim=1)

    #print("s1:", combined_features.shape)

    # Apply 1x1 convolution to merge the channel features
    #combined_features = self.combine_conv(combined_features)
    #combined_features = self.relu(combined_features)

    combined_features = self.combine_pool(combined_features)

    #print("s2:", combined_features.shape)

    # Flatten the output for the fully connected layers
    combined_features = combined_features.view(combined_features.size(0), -1)

    #print("s3:", combined_features.shape)

    # Pass through the fully connected layers
    x = torch.relu(self.fc1(combined_features))
    x = self.fc2(x)

    return x