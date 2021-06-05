import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
from models.small_cnn import SmallCNN
from models.large_cnn import LargeCNN
from models.cam_cnn import CamCNN
from models.pretrained_resnet import ResNetFeatures
from torchvision.models.resnet import ResNet
from utils.data_loaders import denormalize

def visualise_model_predictions(model, loader, exp_dir, save=False):
    """
    Function to visualise heatmaps of model predictions
    If model is CamCNN, CAM technique is used
    Else Grad-CAM technique is used
    
    Visulisations are performed for the class the model predicts, regardless of true image label

    Parameters
    ----------
    model : model to visualise
    loader : loader of data to perform visualisations on
    exp_dir : directory to save visulaisations

    Returns
    -------
    None.

    """
    # Set model to cpu and evaluation mode
    model.cpu()
    model.eval()
    # Randomly choose an image to visualise
    images, labels = next(iter(loader))
    choice = np.random.randint(labels.shape[0])
    img = images[choice].unsqueeze(0)
    label = labels[choice]
    
    # Choose visualisation technique depending on model type
    # to obtain raw heatmap
    if isinstance(model, SmallCNN) or isinstance(model, LargeCNN):
        heatmap, prediction = gradcam_heatmap(model, img)
    elif isinstance(model, CamCNN):
        heatmap, prediction = cam_heatmap(model, img)
    elif isinstance(model, ResNet):
        model = ResNetFeatures(model)
        heatmap, prediction = gradcam_heatmap(model, img)
        # TODO: allow for cam_heatmaps visulaisation with ResNets
    else:
        assert False, "Please ensure a valid 'model_type' string is chosen"
        
    print('True label = {}'.format(loader.dataset.classes[label]))
    print('Model prediction = {}'.format(loader.dataset.classes[prediction]))
    
    # Scale heatmap between 0 and 255
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    # upscale heatmap
    heatmap = cv2.resize(heatmap, (256,256))
    
    # Apply color map and merge with x-ray img
    heatmap = cv2.applyColorMap(255-heatmap, cv2.COLORMAP_JET) / 255
    img = denormalize(img.squeeze())
    result = heatmap * 0.5 + img.cpu().numpy() * 0.5

    # Plot X-Ray and heatmap
    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    axs[0].imshow(img)
    axs[1].imshow(result)
    axs[0].set_title('{} X-Ray'.format(loader.dataset.classes[label]))
    axs[1].set_title('Heatmap for \n{} prediction'.format(loader.dataset.classes[prediction]))
    if save:
        plt.savefig(exp_dir + '/heatmap.png', dpi=100)

   
def gradcam_heatmap(model, img):
    """
    Method to produce grad-CAM heatmaps for either
    SmallCNN, LargeCNN, ResNet1fc, of ResNet3fc
    
    Main resources used:
    https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    https://towardsdatascience.com/grad-cam-2f0c6f3807fe 

    Parameters
    ----------
    model : model to produce visualisations
    img : image to visulaise

    Returns
    -------
    heatmap : produced heatmap.
    prediction : prediction made by model on image

    """
    print('Visualising using Grad-CAM technique...\n')
    # Get output and prediction
    output = model(img, hook=True)
    _, prediction = torch.max(output, 1)
    
    # get the gradient of the output with respect to the parameters of the model
    # Do so for predicted output
    output[:,prediction].backward()
    
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()
    
    # weight the channels by corresponding gradients
    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    return heatmap, prediction



def cam_heatmap(model, img):
    """
    Method to produce CAM heatmaps for CamCNN
    
    Main resources used:
    https://github.com/jrzech/reproduce-chexnet/blob/master/visualize_prediction.py
    https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923

    Parameters
    ----------
    model : model to produce visualisations
    img : image to visulaise

    Returns
    -------
    heatmap : produced heatmap.
    prediction : prediction made by model on image

    """
    print('Visualising using CAM technique...\n')
    output = model(img)
    _, prediction = torch.max(output, 1)
    
    # Get features from last conv layer
    features = model.get_feature_maps(img)
    features = features.squeeze().cpu().detach().numpy()

    # pull weights of fc layer (connects global pooled layers to output)
    weights = model.state_dict()['fc1.weight']
    weights = weights.cpu().numpy()
    bias = model.state_dict()['fc1.bias']
    bias = bias.cpu().numpy()
    
    # # Verify extracted weights and features are correct by comparing normal and manually calculated output
    # prob = nn.Softmax(dim=1)
    # features_glob = features.reshape(-1,256,8*8).mean(-1).reshape(-1,256)
    # output = np.sum(features_glob * weights, axis=-1) + bias
    # print('Normal output: {}'.format(prob(model(img))))
    # print('Numpy output:  {}'.format(prob(torch.tensor(output).unsqueeze(0))))

    # Calculate the class activation map
    cam = np.zeros((8,8,1))
    for i in range(0, 8):
      for j in range(0, 8):
        for k in range(0, 256):
          cam[i, j] += features[k, i, j] * weights[prediction, k] # Use weights for predicted output
    cam += bias[prediction]
    return cam, prediction
    