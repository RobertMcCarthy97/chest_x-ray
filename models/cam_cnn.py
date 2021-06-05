import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CamCNN architecture
# Only 1 fc layer to allow heatmapping. Extra conv layer to compensate
class CamCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super(CamCNN, self).__init__()
        # convolutional layer (sees 3x256x256 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x128x128 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 32x64x64 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # convolutional layer (sees 64x32x32 tensor)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # convolutional layer (sees 128x16x16 tensor)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # linear layer (256 -> 2)
        self.fc1 = nn.Linear(256, 2)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.use_dropout = use_dropout
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x))) 
        # Note the maxpooling of the 5th conv layer is somewht redundant
        # due to the global pooling performed immediately after - should be removed!
        
        # Global average pool the feature maps
        x = x.view(-1,256,8*8).mean(-1).view(-1,256)
        if self.use_dropout:
          x = self.dropout(x)
          
        x = self.fc1(x)
        return x
    
    # Return the feature maps before Global avergae pooling to allow CAM visualisations
    def get_feature_maps(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        return x