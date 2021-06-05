import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from models.small_cnn import SmallCNN
from models.large_cnn import LargeCNN
from models.cam_cnn import CamCNN
from models.pretrained_resnet import get_pretrained_resnet

# Function to set plot font sizes
def set_font_sizes(small_size=14, medium_size=16, large_size=18):        
    plt.rc('font', size=large_size)          # controls default text sizes
    plt.rc('axes', titlesize=large_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=large_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)    # legend fontsize
    plt.rc('figure', titlesize=large_size)  # fontsize of the figure title

def get_training_results(model_type, exp_dir, use_dropout, save=False):
    """
    Function to get results from a training run of a model.
    Must specify a directory with relevant training logs and models saved from the main training loop.
    Directory must follow correct 'format'
    Must also specify the correct model_type for that directory.
    Returns the model weights with the lowest validation loss.

    Parameters
    ----------
    model_type : Type of model to be evaluated
    exp_dir : Directory previously trained models and logs were saved to
    use_dropout : whether model was trained with dropout (need to know when reloading ResNetfc3)

    Returns
    -------
    model : Model with best validation loss

    """
    # Obtain training logs
    results = np.genfromtxt(exp_dir + '/data.csv', delimiter=',')

    # set_font_sizes()
    
    # Plot losses from training
    trainline, = plt.plot(results[:,-2], alpha=1, linewidth=2, label='Train loss')
    validline, = plt.plot(results[:,-1], alpha=1, linewidth=2, label='Valid loss')
    
    plt.title('Training Results')
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylim([0,1])
    plt.legend(handles=[trainline,validline])
    if save:
        plt.savefig(exp_dir + '/trainingplot.png', dpi=100, bbox_inches = "tight")
    
    # Get epoch with lowest validation loss
    best_epoch = np.argmin(results[:,-1])
    print('Best Epoch')
    print('----------')
    print('\nEpoch {}, Train loss = {:.3f}, Validation Loss = {:.3f}\n'.format(best_epoch, results[best_epoch,-2], results[best_epoch,-1]))
    
    # Initialise model
    if model_type == 'SmallCNN':
        model = SmallCNN(use_dropout=use_dropout)
    elif model_type == 'LargeCNN':
        model = LargeCNN(use_dropout=use_dropout)
    elif model_type == 'CamCNN':
        model = CamCNN(use_dropout=use_dropout)
    elif model_type == 'ResNet3fc':
        model = get_pretrained_resnet(dropout_trainable_layers=use_dropout, extra_fc_layers=True)
        # Here model structure depends on whethr dropout was used in training (hence inclusion of use_dropout)
    elif model_type == 'ResNet1fc':
        model = get_pretrained_resnet()
    else:
        assert False, "Please ensure a valid 'model_type' string is chosen"
        
    # Load in weights from lowest validation loss epoch
    model_path = exp_dir + '/model_epoch{}.pt'.format(best_epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
    return model

def get_saved_model(model_type):
    """
    Function to obtain one of the models from the main experiments of the paper.
    Simply specify which model_type is desired and the trained model is returned.
    This is the model which performed best on the validation data for each architecture (i.e. one used for testing)

    Parameters
    ----------
    model_type : Type of model wished to be obtained

    Returns
    -------
    returns the requested model

    """
    # Ensure requested model_type is valid
    valid_models = ['SmallCNN', 'LargeCNN', 'CamCNN', 'ResNet3fc', 'ResNet1fc']
    if model_type not in valid_models:
        assert False, "Please ensure a valid 'model_type' string is chosen"
    
    # Set directory in which trained model is saved
    exp_dir = 'experiments/' + model_type.lower()
    # All models of main experiment were trained with dropout (where applicable)
    use_dropout=True
    # Get trained model
    return get_training_results(model_type, exp_dir, use_dropout)

def evaluate_model(model, test_loader, exp_dir, save=False):
    """
    Function to evaluate a model on the test data.
    Will print confusion matrix and eval measures for a threshold of 0.5.
    Will print AUC curve.

    Parameters
    ----------
    model : model to be tested
    test_loader : data loader to test model on
    exp_dir : directory to save plots in

    Returns
    -------
    None.

    """
    
    # Use gpu for accleration if available
    test_on_gpu = torch.cuda.is_available()
    if test_on_gpu:
        print('Using GPU')
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()   # To calc loss on test data
    probs = nn.Softmax(dim=1)           # To calc model output probabilities
    
    # no.grad() as no need to track gradients
    with torch.no_grad():
        # Set model to evaluation mode
        model.eval()
        # Count relevant stats
        test_imgs = 0.0
        test_loss = 0.0
        test_acc = 0.0
        all_predictions = []
        all_targets = []
        output_probs = []
        # Iterate through all data
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            if test_on_gpu:
                images, labels = images.cuda(), labels.cuda() # Move to gpu if being used
                
            # Get output and calculate accuracy and loss
            output = model(images)
            _, predictions = torch.max(output, 1) # Predict based on largest output probability (i.e., threshold of 0.5)
            accuracy = np.mean(predictions.cpu().detach().numpy() == labels.cpu().detach().numpy())*100
            loss = criterion(output, labels)
            
            # Cumulate stats
            test_loss += loss.item()*images.size(0)
            test_acc += accuracy*images.size(0)
            test_imgs += images.size(0)
            # Append stats
            output_probs += [probs(output).cpu().numpy()] # Convert outputs to true probabilities and append
            all_predictions += [predictions.cpu().numpy()]
            all_targets += [labels.cpu().numpy()]
    
    # Reshape stats
    output_probs = np.array(output_probs).reshape((-1,2))
    all_predictions = np.array(all_predictions).reshape(-1)
    all_targets = np.array(all_targets).reshape(-1)
    
    # calculate average results
    test_loss = test_loss/test_imgs
    test_acc = test_acc/test_imgs
    TPs, FPs, TNs, FNs = get_confusion_matrix(all_predictions, all_targets)
    
    print('Test Loss: {:.3f}'.format(test_loss))
    
    # Plot Confusion matrix
    cf_matrix = np.array([[TPs,FPs],[FNs,TNs]])
    labels = ['TP','FP','FN','TN']
    categories = ['PNEUMONIA', 'NORMAL']
    title = 'Confusion Matrix (threshold = 0.5)'
    make_confusion_matrix(cf_matrix, exp_dir, 
                          group_names=labels,
                          categories=categories, 
                          cmap='binary', title=title, save=save)
    # Plot AUC
    plot_AUC(output_probs, all_targets, exp_dir, save=save)

# Function to calculate confusion matrix stats from provided model predictions and true labels
def get_confusion_matrix(predictions, targets):
    positive_pred_idx = predictions == 1.0
    negative_pred_idx = predictions == 0.0
    TPs = np.sum(predictions[positive_pred_idx] == targets[positive_pred_idx])
    FPs = np.sum(predictions[positive_pred_idx] != targets[positive_pred_idx])
    TNs = np.sum(predictions[negative_pred_idx] == targets[negative_pred_idx])
    FNs = np.sum(predictions[negative_pred_idx] != targets[negative_pred_idx])
    return TPs, FPs, TNs, FNs

# Function to plot ROC and calculate AUC
# Provide models output probabilities and corresponding targets
def plot_AUC(output_probs, targets, exp_dir, save=False):
    # placeholder for TP and FP rates at each tested threshold
    TPRs = []
    FPRs = []
    # Calculate for 1000 threshold evenly spaced between 0 and 1
    thresholds = np.linspace(0,1,num=1000)
    # For each threshold obtain data
    for threshold in thresholds:
        # Get predictions for given threshold 
        predictions = (output_probs[:,1] > threshold) * 1
        # Get number of TPs, FPs etc for these predictions
        TPs, FPs, TNs, FNs = get_confusion_matrix(predictions, targets)
        # Calculate True Positive and False Positive rates
        TPRs += [TPs / (TPs + FNs)]
        FPRs += [FPs / (TNs + FPs)]
    # Calculate AUC of ROC
    AUC = -1 * np.trapz(y=TPRs, x=FPRs)
    print('AUC = {:.3f}'.format(AUC))
    
    # Plot ROC
    plt.figure(2)
    # set_font_sizes()
    model, = plt.plot(FPRs, TPRs, label='AUC = {:.3f}'.format(AUC))
    random, = plt.plot([0,1],[0,1], color='grey', linestyle='dashed', label='')
    plt.title('ROC curve')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(handles=[model])
    if save:
        plt.savefig(exp_dir + '/AUCplot.png', dpi=100, bbox_inches = "tight")
    plt.show()

# Function to visually display confusion matix, along with some stats
# Credit: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
def make_confusion_matrix(cf,
                          exp_dir,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          save=False):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    ax:           the axis to plot on
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    plt.figure(1)
    
    # set_font_sizes()
    
    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
        
    if save:
        plt.savefig(exp_dir + '/confusionmat.png', dpi=100, bbox_inches = "tight")
        
    plt.show()
