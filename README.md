# CS6910-Assignment-2

Welcome to the CS6910 Assignment 2 repository! This assignment focuses on deep learning techniques for image classification and hyperparameter optimization using PyTorch and WandB. Below is an overview of the folder structure and contents of this repository.

## Folder Structure

- **Part-A**:
  - `Question_1.ipynb`: This notebook contains the implementation of a Convolutional Neural Network (CNN) model in PyTorch for image classification using the iNaturalist dataset. It offers customizable parameters such as the number of convolution filters, activation functions (ReLU, GELU, SiLU, Mish), filter organization, batch normalization, and dropout. The model architecture includes five convolutional layers followed by max-pooling layers.
  - `Question_2.ipynb`: This notebook demonstrates neural network training with hyperparameter optimization using PyTorch and WandB. It sets up a hyperparameter sweep, defines parameters like the number of filters, fully connected layer size, activation function, batch normalization, and dropout rate, and performs training and validation epochs.
  - `Question_3.ipynb`: This notebook trains the CNN model using the best hyperparameters detected from the Bayesian Sweep in Question 2 and evaluates its accuracy on the test dataset. Additionally, it randomly samples 10 images from the test dataset and makes predictions using the learned parameters.
  - `train_parta.py`: The main training script that consolidates the functionalities from the notebooks above and logs various metrics in WandB with increasing epochs.

- **Part-B**:
  - `Question_3.ipynb`: This notebook showcases transfer learning for image classification using a pre-trained ResNet-50 model in PyTorch. It freezes all layers except the final fully connected layer and adapts it for a specific classification task with 10 output classes.
  - `train_partb.py`: The main training script for Part B, which includes the transfer learning implementation and metric logging with WandB.

## Usage

1. Start by exploring the `Part-A` or `Part-B` folder based on your interest.
2. Execute the Jupyter notebooks (`*.ipynb`) or Python scripts (`*.py`) to run the code and follow the instructions provided within each file.
3. In most of the codes in this repository, the dataset was loaded onto my Kaggle account and then the codes were run, and hence you would see the command line ```dataset = ImageFolder('/kaggle/input/dataset/inaturalist_12K/train', transform=train_transform)``` (for training) and ``` test_dataset = ImageFolder('/kaggle/input/dataset/inaturalist_12K/val', transform=vanilla_transform)``` (for test) in many places. In case you want to run the code, please modify the line accordingly and then plug in your dataset root directories and then run the codes.
4. If you have any further questions in any part of the assignment, please feel free to contact me at ce22s003@smail.iitm.ac.in (+919083782161).
