import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_pretrained_resnet(freeze_params=True, dropout_trainable_layers=False, extra_fc_layers=False):
    """
    Function to get a pretrianed ResNet50 for transfer learning training.
    ResNet is used as a feature extractor and only the newly added fc layers are optimized (if freeze_params=True)

    Parameters
    ----------
    freeze_params : boolean, optional
        Freezes the conv layer parameters of the pretrained ResNet if True. The default is True.
    dropout_trainable_layers : boolean, optional
        Apply dropout to fc layers if True. The default is False.
        Does not apply when extra_fc_layers=False
    extra_fc_layers : boolean, optional
        add 3 fc layers if True. Else add 1 fc layer. The default is False.

    Returns
    -------
    model : ResNet50 pretrained on ImageNet.

    """
    # Get pretrained model
    model = models.resnet50(pretrained=True)
    
    # Freeze conv layer parameters if specified
    if freeze_params:
      for param in model.parameters():
        param.requires_grad = False

    # Now replace old fc layer with new fc layer(s)
    num_ftrs = model.fc.in_features
    
    # If no extra, just add a single fc layer
    if not extra_fc_layers:
        model.fc = nn.Linear(num_ftrs, 2)
    # Else add 3 fc layers
    else:
        # Add dropout if specified
        if dropout_trainable_layers:
          model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2048, 2)
          )
        # Else no dropout
        else:
          model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2)
          )
    
    return model

class ResNetFeatures(nn.Module):
    """
    This class is used to allow grad-cam visualisations be used on a pneumonia trained ResNet.
    Slightly over complicated process as this feature was implemented retrospactively.
    
    Splits the ResNet into components to allow Feature maps be accessed and gradients be hooked
    
    Grad-cam resources used:
        https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
        https://towardsdatascience.com/grad-cam-2f0c6f3807fe 
    """
    def __init__(self, resnet50):
        super(ResNetFeatures, self).__init__()
        # Assign pneumonia trained model
        self.resnet50 = resnet50
        # Unfreeze params to allow grad-cam visulaisations
        for param in resnet50.parameters():
            param.requires_grad = True
        
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(*list(self.resnet50.children())[:-2])
        
        # Recreate global average pooling layer after final conv layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # After pooling use the classifier that has been trained for pneumonia predictions
        self.classifier = self.resnet50.fc
        
        # placeholder for the gradients
        self.gradients = None
    
    # function to hook gradients (grad-cam)
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x, hook=True):
        # Get feature maps from last conv layer
        x = self.features_conv(x)
        # Hook gradients
        if hook:
            # Register gradient hook at final conv layer before global pooling for grad_cam visualisations
            h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avgpool(x)
        # Flatten and pass x through fc layers 
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x
    
    # method to return hooked gradients (grad-cam)
    def get_activations_gradient(self):
        return self.gradients
    
    # method to obtain feature maps of final conv layer before maxpooling (grad-cam)
    def get_activations(self, x):
        return self.features_conv(x)