!pip install wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import models
import wandb

wandb.login(key='4734e60951ce310dbe17484eeeb5b3366b54850f')
wandb.init(project='CS6910_assignment_2', entity='sumanta_roy')

batch_size=32
num_epochs=5
learning_rate=1e-3

model = models.resnet50(pretrained=True)

# Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False

# Replacing the final fully connected layer with a new one for 10 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Changing 10 to the number of classes in my dataset

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vanilla_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = ImageFolder('/kaggle/input/dataset/inaturalist_12K/train', transform=train_transform)
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, shuffle=True, stratify=dataset.targets)
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

test_dataset = ImageFolder('/kaggle/input/dataset/inaturalist_12K/val', transform=vanilla_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if torch.cuda.is_available():
    model=model.cuda()
    criterion=criterion.cuda()

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        train_loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    print(f'At end of epoch [{epoch+1}/{num_epochs}], training loss: {train_loss.item():.4f}')  

    # Validation phase
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            val_loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'At end of epoch [{epoch+1}/{num_epochs}], validation loss: {val_loss.item():.4f}, validation acuracy: {val_accuracy:.2f}%')

    wandb.log({'train_loss': train_loss.item(),'validation_loss': val_loss.item(), 'val_accuracy': val_accuracy, 'epoch': epoch+1})


print('Finished Training')
# Test phase
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = 100 * correct / total
print(f'Test accuracy: {test_accuracy:.2f}%')

wandb.finish()
