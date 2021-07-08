# Chest X-ray

<p align="center">
  <img width="600" src="https://github.com/RobertMcCarthy97/chest_x-ray/blob/main/experiments/resnet1fc/heatmap.png">
</p>

In this project, CNNs are trained to detect pneumonia in chest x-rays.

The project report can be found in the following pdf file:
'COMP47650 Deep Learning Project: Pneumonia Chest X-Ray Classification.pdf'

### Notebooks

There are two jupyter notebooks provided:

1. reproduce_results.ipynb
	Run this notebook to reproduce the results of the report
	by retraining then evaluating a model
	
2. evaluate_trained_models.ipynb
	Run this notebook to evaluate the saved models
	whose results were presented in the report.
	WARNING: models have not been saved to github repo due to large file size restrictions
	
Further details can be found in the individual notebooks.


### Directory Structure

- utils (folder: contains utility functions)
	- data_loaders.py (contains functions related to obtaining the data and data loaders)
	- training.py (contains functions used for training models)
	- evaluation.py (contains functions used to evaluate models on test data)
	- visualisations.py (implements grad-cam and cam techniques for visualisations)
	
- models (folder: contains all model implementations)
	- small_cnn.py (contains SmallCNN model class)
	- large_cnn.py (contains LargeCNN model class)
	- cam_cnn.py (contains CamCNN model class)
	- pretrained_resnet.py (contains ResNet related function and class)
	
- experiments (folder: contains data from the main experiments)
	- small_cnn (contains saved model and data of small_cnn)
	- cam_cnn (contains saved model and data of cam_cnn)
	- resnet3fc (contains saved model and data of resnet3fc)
	- resnet1fc (contains saved model and data of resnet1fc)
	- readme.txt (describes results saved for each model)
