from torch import nn
from torchvision.models import resnet50
import torch

class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        #x = torch.softmax(x, dim=-1)
        return x


class Backbone(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

        """
        if weight_path:
            self.load_radimagenet_weights(weight_path)
        """
                        
    def forward(self, x):
        x = x.squeeze(dim=2)
        return self.backbone(x)

    def load_radimagenet_weights(self, weight_path):
        # Load the weights from the .pt file
        radimagenet_weights = torch.load(weight_path)

        # Apply the weights
        self.backbone.load_state_dict(radimagenet_weights, strict=False)

        # Make the parameters non-trainable
        for param in self.backbone.parameters():
            param.requires_grad = False

class ThreeSliceClassifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear1 = nn.Linear(3 * 2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class ThreeSliceBackbone(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

        if weight_path:
            self.load_radimagenet_weights(weight_path)
                        
    def forward(self, x):
        batch_size, channels, slices, height, width = x.shape
        output_slices = []
        for i in range(slices):
            slice_i = x[:, :, i, :, :]
            output = self.backbone(slice_i)
            output_slices.append(output)

        output = torch.stack(output_slices, dim=2)

        return output

    def load_radimagenet_weights(self, weight_path):
        # Load the weights from the .pt file
        radimagenet_weights = torch.load(weight_path)
    
        self.backbone.load_state_dict(radimagenet_weights, strict=False)


class MultiChannelClassifier(nn.Module):
    def __init__(self, num_class, num_channels):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear1 = nn.Linear(num_channels * 2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MultiChannelBackbone(nn.Module):

    # Train a set of weights for each input channel separately

    def __init__(self, num_channels, weight_path):
        super().__init__()
        self.num_channels = num_channels
        self.models = nn.ModuleList([self._create_model(weight_path) for _ in range(num_channels)])

    def _create_model(self, weight_path):
        # Initialize a ResNet50 model
        model = resnet50(pretrained=False)
        encoder_layers = list(model.children())
        backbone = nn.Sequential(*encoder_layers[:9])

        self.load_radimagenet_weights(backbone, weight_path)

        return backbone

    def forward(self, x):

        x = x.squeeze(dim=2)

        # Process each channel through its corresponding ResNet50 model
        outputs = [model(x[:, i:i+1, :, :].expand(-1, 3, -1, -1)) for i, model in enumerate(self.models)]

        # Concatenate the outputs along the channel dimension
        out = torch.cat(outputs, dim=1)

        return out

    def load_radimagenet_weights(self, model, weight_path):
        # Load the weights from the .pt file
        radimagenet_weights = torch.load(weight_path)

        # Apply the weights
        model.load_state_dict(radimagenet_weights, strict=False)

    
class SliceProcessor(nn.Module):
    def __init__(self, pretrained_model):
        super(SliceProcessor, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, x):
        # x is a batch of 3D volumes: [batch_size, channels, depth, height, width]
        batch_size, channels, depth, height, width = x.shape

        # Process each slice separately and store the features
        processed_slices_per_channel = []
        slice_features = []
        for i in range(channels):
            for j in range(depth):
                slice_i = x[:, i, j, :, :]  # Get the ith slice
                img_tensor = slice_i.unsqueeze(1)
                img_tensor_3_channels = img_tensor.repeat_interleave(3, dim=1)
                features_i = self.pretrained_model(img_tensor_3_channels)  # Process ith slice
                slice_features.append(features_i)
            processed_slices_per_channel.append(torch.stack(slice_features, dim=2))
            slice_features=[]

        return processed_slices_per_channel 

class AttentionModel(nn.Module):
    def __init__(self, preprocess_model):
        super(AttentionModel, self).__init__()
        
        # Define the attention mechanism
        self.backbone_model = preprocess_model
        self.attention_weights_1 = nn.Sequential(
            nn.Conv3d(2048, 1, kernel_size=(1, 1, 1)),  # Learnable weights for each depth
            nn.Softmax(dim=2)  # Apply softmax along the depth dimension (dim=2)
        )
        
        # Define the downsampling layer
        self.downsampling = nn.Sequential(
            nn.Conv3d(2048, 1024, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 1024, kernel_size=(3,3,3)),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            # nn.Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1,1,1)),
            # nn.BatchNorm3d(128),
            # nn.ReLU(),
        )
        self.attention_weights_2 = nn.Sequential(
            nn.Conv3d(768, 1, kernel_size=(1, 1, 1)),  # Learnable weights for each depth
            nn.Softmax(dim=1)  # Apply softmax along the depth dimension (dim=2)
        )
        self.downsampling_2 = nn.Sequential(
            nn.Conv3d(768, 128, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        
        # Define the classification head
        self.classifier = nn.Sequential(
            nn.Linear(128*28*1*2, 256),  # Adjust input size and hidden layers as needed
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),  # Adjust input size and hidden layers as needed
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3),  # num_classes is the number of classes
        )

    def forward(self, x):

        # process images with 2d model (per slice)
        process_radimagenet_vectors = self.backbone_model(x)
        t2w_vectors = process_radimagenet_vectors[0]
        adc_vectors = process_radimagenet_vectors[1]
        diff_vectors = process_radimagenet_vectors[2]

        # Apply the attention mechanism to compute attention weights
        attention_weights_1_t2w = self.attention_weights_1(t2w_vectors)
        attention_weights_1_adc = self.attention_weights_1(adc_vectors)
        attention_weights_1_diff = self.attention_weights_1(adc_vectors)

        # Apply attention weights to the input tensor
        weighted_x_1_t2w = t2w_vectors * attention_weights_1_t2w
        weighted_x_1_adc = adc_vectors * attention_weights_1_adc
        weighted_x_1_diff = diff_vectors * attention_weights_1_diff

        # Downsample the tensor
        downsampled_x_t2w = self.downsampling(weighted_x_1_t2w)
        downsampled_x_adc = self.downsampling(weighted_x_1_adc)
        downsampled_x_diff = self.downsampling(weighted_x_1_diff)
        fused_tensor_concat = torch.cat((downsampled_x_t2w, downsampled_x_adc, downsampled_x_diff), dim=1)
        # print('shape', fused_tensor_concat.shape)

        attention_weights_2 = self.attention_weights_2(fused_tensor_concat)

        weighted_x_2 = fused_tensor_concat * attention_weights_2
        weighted_x_3 = self.downsampling_2(weighted_x_2)

        # Flatten the tensor for the classifier
        flattened_x = weighted_x_3.view(weighted_x_3.size(0), -1)

        # Forward pass through the classifier
        logits = self.classifier(flattened_x)

        return logits