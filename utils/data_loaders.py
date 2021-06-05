import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import time

def create_loaders(data_dir, augment_data, weight_samples, batch_size, use_og_split, num_workers=2):
    """
    Creates the data loaders for training, validation and testing

    Parameters
    ----------
    data_dir : String
        Directory the chest x-ray is stored in
    augment_data : boolean
        Will augment training data if True
    weight_samples : boolean
        Minority class will be oversampled if True
    batch_size : int
        size of batches to use in mini-batch gradient descent
    use_og_split : boolean
        Maintains the default/original train/val/test split if True.
    num_workers : int, optional
        Number of workers to use to load data. The default is 2.

    Returns
    -------
    The training, validation, and test loaders
        
    NOTE: The alternative split here is different to that discussed in the paper 
    and has not been fully tested
    """
    # Get transforms
    train_transform, test_transform = get_transforms(augment=augment_data)
    
    # Get training data
    train_dir = data_dir + '/train'
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # If using original data split then proceed as normal
    if use_og_split:
        valid_dir = data_dir + '/val'
        test_dir = data_dir + '/test'
        train_idx, valid_idx, test_idx = slice(None), slice(None), slice(None)
        supress_idx = None
        valid_sampler = None
        test_sampler = None
        valid_shuffle = True # See note below on shuflling
        test_shuffle = True
    # Else split original training data into train/val/test and ignore original val and test data
    # This split is achieved with SubsetRandomSampler
    else:
        print('\nWARNING: This alternative split of the data has not been fully tested\n')
        valid_dir = data_dir + '/train'
        test_dir = data_dir + '/train'
        # Get indices of train/val/test images based on split ratios
        train_idx, valid_idx, test_idx = get_split_idxs(train_data, split=[0.8,0.1,0.1])
        supress_idx = valid_idx + test_idx
        # Create samplers for split
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_shuffle = False
        test_shuffle = False
        
    # Get valid and test data
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # Weight training samples to oversample minority class if specified
    if weight_samples:
        train_sampler = get_weighted_sampler(train_data, supress_idx)
        train_shuffle = False
    # Else ensure training data is shuffled if original split is being used
    else:
        if use_og_split:
            train_sampler = None
            train_shuffle = True
        else:
            train_sampler = SubsetRandomSampler(train_idx)
            train_shuffle = False
    
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                sampler=train_sampler, shuffle=train_shuffle, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                                sampler=valid_sampler, shuffle=valid_shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                sampler=test_sampler, shuffle=test_shuffle, num_workers=num_workers)
    
    # Note: validation and test data are only shuffled so PNEUMONIA images can be easily accessed for visualisations
    # Without shuffle we have to iterate through half the data to reach PNEUMONIA images
    # Shuffling makes no difference to evaluations
    
    # print info aboput the split of the data
    print_data_info(train_data, valid_data, test_data, train_idx, valid_idx, test_idx)
    
    return train_loader, valid_loader, test_loader


def get_transforms(augment=True):
    """
    Returns the transforms to be used on the data
    
    """
    # No augmentation is used for test data transform
    test_transform = transforms.Compose([transforms.Resize((268,268)),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    # If specified, training data is augmented with thhe transforms seen below
    if augment:
      print('Augmenting training data...')
      train_transform = transforms.Compose([transforms.Resize((268,268)),
                                          transforms.RandomAffine(20, translate=(0.1,0.1)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.CenterCrop(256),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    else:
      train_transform = test_transform
    
    return train_transform, test_transform


def get_weighted_sampler(train_data, suppress_idx=None):
    """
    This function takes in the training data and returns a sampler which
    weights images based on the prevalence of their class in the data.
    I.e., weights images so that minority class is oversampled
    This ensures rouhghly an equal number of NORMAL and PNEUMONIA images are sampled each epoch

    """
    print('Weighting sampling of training data...')
    # Count images in each class
    class_sample_counts = []
    for i in range(len(train_data.classes)):
      class_sample_counts += [np.sum(np.array(train_data.targets) == i)]
    class_sample_counts = np.array(class_sample_counts)
    
    # Weight images based on the count of their class
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_data.targets]
    
    # If using the alternative data split, then we must ensure no validation or test images are used in training
    if suppress_idx != None:
        # Set weights of validation and test data to 0 so none are sampled
        samples_weights[suppress_idx] = 0.0

    # Create and return sampler
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)
    return sampler


# Returns randomlly selected indices for alternative train/val/test split
# split=[train_proportion,valid_proportion,test_proportion]
def get_split_idxs(data, split=[0.8,0.1,0.1]):
    num_train = len(data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_split = int(np.floor(split[0] * num_train))
    val_split = int(np.floor( (split[0]+split[1]) * num_train))
    train_idx, valid_idx, test_idx = indices[:train_split], indices[train_split:val_split], indices[val_split:]
    return train_idx, valid_idx, test_idx

# prints info about the data split being used
def print_data_info(train_data, valid_data, test_data, train_idx, valid_idx, test_idx):
    train_normal = np.sum(np.array(train_data.targets)[train_idx] == 0)
    train_pneu = np.sum(np.array(train_data.targets)[train_idx] == 1)
    valid_normal = np.sum(np.array(valid_data.targets)[valid_idx] == 0)
    valid_pneu = np.sum(np.array(valid_data.targets)[valid_idx] == 1)
    test_normal = np.sum(np.array(test_data.targets)[test_idx] == 0)
    test_pneu = np.sum(np.array(test_data.targets)[test_idx] == 1)
    print('\nDetails of train/val/test split:\n')
    print('      Normal | PNEU | TOTAL')
    print('TRAIN:  {} | {} | {}'.format(train_normal, train_pneu, train_normal + train_pneu))
    print('VALID:     {} |    {} |   {}'.format(valid_normal, valid_pneu, valid_normal + valid_pneu))
    print('TEST :   {} |  {} |  {}'.format(test_normal, test_pneu, test_normal + test_pneu))

# Performs a quick inspection of a dataloader by sampling a batch
def quick_loader_test(loader):
    images, labels = next(iter(loader))
    img = images[0]
    label = labels[0]
    print('Batch size: {}'.format(labels.shape[0]))
    for i in range(len(loader.dataset.classes)):
      print('{} images: {}'.format(loader.dataset.classes[i], torch.sum(labels == i)))
    print('Image shape: {}'.format(img.detach().numpy().shape))
    img = denormalize(img)
    plt.imshow(img)
    plt.title(loader.dataset.classes[label])

# Runs through all data in loader and prints info
def full_loader_test(loader):
    t0 = time.time()
    batches = 0
    zeros = 0
    ones = 0
    print('Iterating through all data...')
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        batches += 1
        zeros += labels.shape[0] - torch.sum(labels).item()
        ones += torch.sum(labels).item()
    print('Total batches: {}'.format(batches))
    print('Total NORMAL: {}'.format(zeros))
    print('Total PNEUMONIA: {}'.format(ones))
    t1 = time.time()
    print('Time taken: {:.2f} minutes'.format(int(t1-t0)/60))

# Denoramlizes images obtained from loader to allow visualisation    
def denormalize(img):
    img = img.permute(1,2,0)
    mean = torch.FloatTensor([0.5, 0.5, 0.5])
    std = torch.FloatTensor([0.5, 0.5, 0.5])
    img = img*std + mean
    img = np.clip(img,0,1)
    return img