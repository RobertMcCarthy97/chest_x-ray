import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the SmallCNN architecture
class SmallCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super(SmallCNN, self).__init__()
        # convolutional layer (sees 3x256x256 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x128x128 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 32x64x64 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # convolutional layer (sees 64x32x32 tensor)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (128*16*16 -> 1024)
        self.fc1 = nn.Linear(128*16*16, 1024)
        # linear layer (1024 -> 2)
        self.fc2 = nn.Linear(1024, 2)
        
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.use_dropout = use_dropout
        
    # function to hook gradients (grad-cam)
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=False):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        if hook:
            # Register gradient hook at final conv layer before maxpooling for grad_cam visualisations
            h = x.register_hook(self.activations_hook)
        x = self.pool(x)
        # flatten image input (sees 128x16x16 tensor)
        x = x.view(-1, 128 * 16 * 16)
        if self.use_dropout:
          x = self.dropout(x)
          
        x = F.relu(self.fc1(x))
        if self.use_dropout:
          x = self.dropout(x)
          
        x = self.fc2(x)
        return x
    
    # method to return hooked gradients (grad-cam)
    def get_activations_gradient(self):
        return self.gradients
    
    # method to obtain feature maps of final conv layer before maxpooling (grad-cam)
    def get_activations(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return x