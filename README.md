# CS6910-Assignment-2

There are two folders in the repository 'Part-A' and 'Part-B', which contain the necessary sub-codes (codes for each question whevenever needed), as well as the master-codes in the Python scripts 'train_parta.py' and 'train_partb.py'. Let us go over the contents one by one:

## Folder Part-A:

- /Question_1.ipynb: This is the python notebook containing the solution to Question 1 of Part A. The small code defines a Convolutional Neural Network (CNN) model in PyTorch for image classification on the iNaturalist dataset that we received as part of the question. The code is very user-friendly and allows customization of parameters like the number of convolution filters for the first convolution layer, choice of activation functions (ReLU, GELU, SiLU, Mish) for the convolution layers, organization of filters in layers (same, double, halve), batch normalization, and dropout. The dropout is applied after the final MaxPool layer to the flattened input to the fully connected layer. The CNN model architecture consists of five convolutional layers followed by max-pooling layers. The forward method executes the layers sequentially, applying activations, pooling, and dropout as specified, and includes custom activation functions like Mish.
- /Question_2.ipynb: This code is for training a neural network using PyTorch and WandB (Weights and Biases) for hyperparameter optimization. It starts by installing the necessary packages and importing the required libraries. Then, it sets up a hyperparameter sweep using WandB, defining the parameters to be tuned, such as the number of filters, filter organization, fully connected layer size, data augmentation, activation function, batch normalization, and dropout rate. The neural network training function is defined, which creates a CNN model based on the specified hyperparameters, sets up the optimizer and loss function, and performs training and validation epochs. The main function initializes the WandB run, sets up the hyperparameters based on the sweep configuration, defines data transformations, loads the dataset, and runs the neural network training function using the hyperparameter values from the sweep. Finally, it uses WandB's agent method to run the main function multiple times with different hyperparameter combinations as specified in the sweep, facilitating automated hyperparameter optimization and logging the results with WandB.
- /Question_3.ipynb: This code is mainly to train the CNN model using the best hyperparameters detected from the Bayesian Sweep from question 2, and then using this set of hyperparaeter to find the accuracy on the test dataset. Finally, the code also randomly samples 10 images from the test dataset and then predicts them using the learned parameters and prints the images along with the predictions.
- /train_parta.py: This is the main training file, which culminates all of the above.

## Folder Part-B:

- /Question_3.ipynb: This code snippet demonstrates a common approach to transfer learning for image classification using a pre-trained ResNet-50 model in PyTorch. Transfer learning is employed by loading the pre-trained ResNet-50 model from ```torchvision.models``` and freezing all its layers except the final fully connected layer. The final layer is replaced with a new one suitable for the specific classification task, which involves 10 output classes in this case. The data preprocessing includes defining transformations for training and testing data, such as resizing, normalization, and converting to tensors. 


