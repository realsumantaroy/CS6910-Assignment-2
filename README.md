# CS6910-Assignment-2

There are two folders in the repository 'Part-A' and 'Part-B', which contain the necessary sub-codes (codes for each question whevenever needed), as well as the master-codes in the Python scripts 'train_parta.py' and 'train_partb.py'. Let us go over the contents one by one:

## Folder Part-A:

- Question_1ipynb: This is the python notebook containing the solution to Question 1 of Part A. The small code defines a Convolutional Neural Network (CNN) model in PyTorch for image classification on the iNaturalist dataset that we received as part of the question. The code is very user-friendly and allows customization of parameters like the number of convolution filters for the first convolution layer, choice of activation functions (ReLU, GELU, SiLU, Mish) for the convolution layers, organization of filters in layers (same, double, halve), batch normalization, and dropout. The dropout is applied after the final MaxPool layer to the flattened input to the fully connected layer. The CNN model architecture consists of five convolutional layers followed by max-pooling layers. The forward method executes the layers sequentially, applying activations, pooling, and dropout as specified, and includes custom activation functions like Mish.

