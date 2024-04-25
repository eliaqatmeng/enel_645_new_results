import os
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import torch.nn.functional as F


# Define the output file path
output_file = 'testing/vgg19/brain_tumor_classification_VGG19_trained.txt'

# Redirect standard output to the output file
sys.stdout = open(output_file, 'w')

# Check if CUDA GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Function to get a subset of dataset indices - In case it is taking too long to train in the cluster
def get_subset_indices(dataset, fraction=1):
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return indices

# Define the root directory containing the "brain tumor" and "healthy" folders
root_dir = r"C:\Users\Gigabyte\Downloads\Brain Tumor Data Set\Brain Tumor Data Set"

# Transform dataset input to match inputs for ResNet-101 model
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load datasets
full_dataset = ImageFolder(root=root_dir, transform=data_transform)

# Get subset indices
subset_indices = get_subset_indices(full_dataset)

# Create subset dataset
subset_dataset = Subset(full_dataset, subset_indices)

# Split dataset into train, validation, and test sets
num_samples = len(subset_dataset)
train_size = int(0.7 * num_samples)  
val_size = int(0.15 * num_samples)   
test_size = num_samples - train_size - val_size  

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size, test_size])

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained VGG19 model
model = vgg19(pretrained=True)

# Modify the classifier to match the number of output classes (assuming 2 output classes)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 2)

# Move the model to the appropriate device (CPU or GPU)
model = model.to(device)

# Define a loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
all_predicted_labels = []
all_true_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()

        all_predicted_labels.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
test_conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

# Calculate precision, recall, and F1 score for test set
test_precision = precision_score(all_true_labels, all_predicted_labels)
test_recall = recall_score(all_true_labels, all_predicted_labels)
test_f1 = f1_score(all_true_labels, all_predicted_labels)

print(f'Test Loss: {test_loss/len(test_dataloader):.4f}, '
    f'Test Accuracy: {(correct/total)*100:.2f}%, '
    f'Precision: {test_precision:.4f}, '
    f'Recall: {test_recall:.4f}, '
    f'F1 Score: {test_f1:.4f}')




# Load and preprocess the image
image_path = r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\vgg19\data\Brain Tumor\Cancer (2406).jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.resize(image, (224, 224))  # Resize to match VGG19 input size
image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

# Get the predicted class label
with torch.no_grad():
    model.eval()
    outputs = model(image_tensor)
    _, predicted_class_idx = torch.max(outputs, 1)

# Map the predicted class index to the actual class label
class_labels = ['Healthy', 'Brain Tumor']  # Replace with your actual class labels
predicted_class_label = class_labels[predicted_class_idx]

# Print the predicted class label
print(f"Predicted Class Label: {predicted_class_label}")

# Function to compute Grad-CAM
def compute_grad_cam(model, img_tensor, target_class_idx=None):
    model.eval()
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()
    
    output = model(img_tensor)
    
    if target_class_idx is None:
        target_class_idx = output.argmax()

    model.zero_grad()
    output[0, target_class_idx].backward()
    
    gradients = img_tensor.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    last_conv_layer = None
    def hook(module, input, output):
        nonlocal last_conv_layer
        last_conv_layer = output
    model.features[-1].register_forward_hook(hook)  # Assuming the last convolutional layer is model.features[-1]
    
    model(img_tensor)
    heatmap = torch.mean(last_conv_layer, dim=1)
    for i in range(heatmap.size(0)):
        heatmap[i, :, :] *= pooled_gradients[i]

    heatmap = F.relu(torch.sum(heatmap, dim=0)).detach().cpu().numpy()

    return heatmap


# Get the Grad-CAM heatmap
heatmap = compute_grad_cam(model, image_tensor)

# Resize the heatmap to match the image size
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap / np.max(heatmap))

# Apply colormap to the heatmap
heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on the original image
superimposed_img = heatmap_colormap * 0.5 + image * 0.5
superimposed_img = np.uint8(superimposed_img)

# Convert the superimposed image to RGB (cv2 reads images in BGR format)
superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# Display the superimposed image
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()

# Save the superimposed image with class label, predicted class label, and heatmap scale
class_label = "Brain Tumor"  # Change this to the appropriate class label
heatmap_scale = "Relative"  # Change this to "Relative" or "Absolute"

output_image_path = "gradcam_output.png"
cv2.imwrite(output_image_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

print(f"Grad-CAM image saved as {output_image_path}")
print(f"Class Label: {class_label}")
print(f"Heatmap Scale: {heatmap_scale}")