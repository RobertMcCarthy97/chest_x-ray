import torch
import torch.nn as nn
import torch.nn.functional as F


# define the LargeCNN architecture
class LargeCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super(LargeCNN, self).__init__()
        # convolutional layer (sees 3x256x256 image tensor)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # convolutional layer (sees 32x128x128 image tensor)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # convolutional layer (sees 64x64x64 tensor)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # convolutional layer (sees 128x64x64 tensor)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        # convolutional layer (sees 128x32x32 tensor)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # convolutional layer (sees 256x32x32 tensor)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        # convolutional layer (sees 256x16x16 tensor)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        # convolutional layer (sees 512x16x16 tensor)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (512 * 8 * 8 -> 2048)
        self.fc1 = nn.Linear(512*8*8, 2048)
        # linear layer (2048 -> 2)
        self.fc2 = nn.Linear(2048, 2)
        
        # dropout layer (p=0.5)
        self.dropout = nn.Dropout(0.5)
        self.use_dropout = use_dropout

        # placeholder for hooked gradients
        self.gradients = None
        
    
    # function to hook gradients (grad-cam)
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, hook=False):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        if hook:
            print('Hooking gradients')
            # Register gradient hook at final conv layer before maxpooling for grad_cam visualisations
            h = x.register_hook(self.activations_hook)
        x = self.pool(x)
        # flatten image input (sees 512x8x8 tensor)
        x = x.view(-1, 512*8*8)
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
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return x