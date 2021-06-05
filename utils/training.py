import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import csv
from models.small_cnn import SmallCNN
from models.large_cnn import LargeCNN
from models.cam_cnn import CamCNN
from models.pretrained_resnet import get_pretrained_resnet

def create_model_and_optimizer(model_type, use_dropout, use_adam, learning_rate):
    """
    Function to create the model and optimizer for training

    Parameters
    ----------
    model_type : String
        The type of model to be trained. See README for more details
    use_dropout : boolean
        Whether to use dropout in the fc layers of the model.
        Does not apply to ResNetfc1
    use_adam : boolean
        Use adam as optimizer if True. Else SGD is used
    learning_rate : float
        Base learning rate to use with optimizer.

    Returns
    -------
    The chosen model and corresponding optimizer

    """
    # Get model depending on choice
    if model_type == 'SmallCNN':
        model = SmallCNN(use_dropout=use_dropout)
        model_to_optimize = model
    elif model_type == 'LargeCNN':
        model = LargeCNN(use_dropout=use_dropout)
        model_to_optimize = model
    elif model_type == 'CamCNN':
        model = CamCNN(use_dropout=use_dropout)
        model_to_optimize = model
    elif model_type == 'ResNet3fc':
        # Replace ResNet50 fc layer 3 new fc layers 
        model = get_pretrained_resnet(freeze_params=True, dropout_trainable_layers=use_dropout, extra_fc_layers=True)
        model_to_optimize = model.fc # Only optimize fc layers of pretrained ResNet50
        print('Training fc layers only')
    elif model_type == 'ResNet1fc':
        # Replace ResNet50 fc layer 1 new fc layer
        model = get_pretrained_resnet(freeze_params=True)
        model_to_optimize = model.fc
        print('Training fc layers only')
    else:
        assert False, "Please ensure a valid 'model_type' string is chosen"
    
    print('Model details:')
    print('------------')
    print(model)
    count_parameters(model, model_type) # Count trainable parameters in the model
    
    # Assign optimizer
    if use_adam:
      optimizer = optim.Adam(model_to_optimize.parameters(), lr=learning_rate, weight_decay=0.0005)
    else:
      optimizer = optim.SGD(model_to_optimize.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    print('\nOptimizer details:')
    print('----------------')
    print(optimizer)
    
    return model, optimizer

# Function to count trainable parameters of a model    
def count_parameters(model, model_type):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{} trainable parameters: {:,}'.format(model_type, num_params))

    
def train_model(model, optimizer, train_loader, valid_loader, n_epochs, exp_dir):
    """
    This is the main training loop method.

    Parameters
    ----------
    model : The model to be trained
    optimizer : The models optimizer
    train_loader : Training data loader
    valid_loader : Validation data loader
    n_epochs : Number of epochs to train for
    exp_dir : Directory to save models and training logs

    Returns
    -------
    None.

    """
    # Make save directory if does not exist
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    # Train model on gpou if one is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU...')
        model.cuda()
    else:
        print('WARNING: not utilising a GPU')
        
    print('[{}] Beginning Training...'.format(datetime.now()))
    criterion = nn.CrossEntropyLoss()   # Use cross-entropy loss
    valid_loss_min = np.Inf             # Keep track of lowwest validation loss
    train_losses = []                   # Keep track of training loss each epoch
    valid_losses = []                   # Keep track of validation loss each epoch
    
    for epoch in range(0, n_epochs):
        # Placeholders to count epoch losses and no. images used
        train_imgs = 0.0
        valid_imgs = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        
        # Train model
        model.train()
        # Iterate through all data
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda() # Move to gpu if being used
            
            # Zero pre-existing gradients
            optimizer.zero_grad()
            output = model(images)
            # Get loss of output
            loss = criterion(output, labels)
            # backpropagate loss and update weights
            loss.backward()
            optimizer.step()
            
            # Cumulate counts
            train_loss += loss.item()*images.size(0)
            train_imgs += images.size(0)
            
        # Validate model
        # Don't track gradients in order to speed up
        with torch.no_grad():
            # Turn off dropout etc by putting model in eval mode
            model.eval()
            # Iterate through all data
            for i, data in enumerate(valid_loader, 0):
                images, labels = data
                if train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                output = model(images)
                # Get model predictions and calculate accuracy
                # Prediction is output with highest probability
                _, predictions = torch.max(output, 1)
                accuracy = np.mean(predictions.cpu().detach().numpy() == labels.cpu().detach().numpy())*100
                loss = criterion(output, labels)
                
                # Cumulate counts 
                valid_loss += loss.item()*images.size(0)
                valid_acc += accuracy*images.size(0)
                valid_imgs += images.size(0)
        
        # Calculate average losses and accuracy
        train_loss = train_loss/train_imgs
        valid_loss = valid_loss/valid_imgs
        valid_acc = valid_acc/valid_imgs
        # Append average losses
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # print training/validation statistics 
        print('[{}] Epoch: {}, Train Loss: {:.3f} , Valid Loss: {:.3f} , Valid Acc: {:.3f}%'.format(
            datetime.now(), epoch, train_loss, valid_loss, valid_acc))
        
        # Append epoch stats to csv file
        with open(exp_dir + '/data.csv', mode='a') as train_file:
            train_writer = csv.writer(train_file)
            train_writer.writerow([epoch, train_loss, valid_loss])
    
        # save model if validation loss has decreased or every 10 epochs
        if valid_loss <= valid_loss_min or epoch % 10 == 0:
            torch.save(model.state_dict(), exp_dir + '/model_epoch{}.pt'.format(epoch))
            torch.save(optimizer.state_dict(), exp_dir + '/optimizer.pt')
            
        # Assign new lowest validation loss if loss has dropped
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss