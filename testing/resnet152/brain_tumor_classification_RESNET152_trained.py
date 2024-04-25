import os
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet152
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import torch.nn.functional as F

# Define the output file path
output_file = 'NewOutputs/resnet152/brain_tumor_classification_RESNET152_trained.txt'

# Redirect standard output to the output file
sys.stdout = open(output_file, 'w')


# Check if CUDA GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Function to save training checkpoint - saved after each epoch is run
def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch + 1,  # Saving next epoch to start from
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

# Function to get a subset of dataset indices - In case it is taking too long to train in the cluster
def get_subset_indices(dataset, fraction=1):
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return indices

# Function to calculate accuracy per class
def accuracy_per_class(conf_matrix):
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    return class_acc

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, classes, filename=None):
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j+0.5, i+0.5, conf_matrix[i, j], ha='center', va='center', color='orange')

    if filename:
        plt.savefig(filename)  # Save the plot as an image
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()
# Function to plot ROC curve and output AUC
def plot_roc_curve_and_output_auc(labels, probs, filename_png='NewOutputs/resnet152/ROC_curve_resnet152_trained.png', filename_csv='NewOutputs/resnet152/ROC_data_resnet152_trained.csv'):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename_png)
    plt.close()

    # Output AUC data as CSV
    roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
    roc_data.to_csv(filename_csv, index=False)

    print(f'AUC: {roc_auc:.4f}')

# Function to compute Grad-CAM

def compute_grad_cam(model, img_tensor, target_class=None):
    model.eval()
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()
    
    output = model(img_tensor)
    
    if target_class is None:
        target_class = output.argmax()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get the gradients of the output with respect to the feature maps
    gradients = img_tensor.grad
    
    # Get the activations of the last convolutional layer using the forward hook
    activations = None
    def hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    hook_handle = model.layer4[-1].register_forward_hook(hook)  # Assuming layer4 is the last convolutional layer
        
    # Pass a dummy input through the model to trigger the forward hook
    model(img_tensor)
    
    # Remove the forward hook
    hook_handle.remove()
    
    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight the activations by the corresponding gradients
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # Compute the heatmap by averaging the weighted activations along the channels
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # ReLU on the heatmap
    heatmap = F.relu(heatmap)
    
    return heatmap

def save_grad_cam_as_image(grad_cam, original_img, predicted_class, actual_class, filename=None):
    # Normalize the grad_cam for proper visualization
    grad_cam = grad_cam.cpu().numpy()
    grad_cam -= grad_cam.min()
    grad_cam /= grad_cam.max()
    grad_cam = np.uint8(255 * grad_cam)
    
    # Resize grad_cam to the size of original image
    heatmap = cv2.resize(grad_cam, (original_img.shape[1], original_img.shape[0]))
    
    # Apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * 0.25 + original_img
    
    # Plot Grad-CAM with predicted class and actual class
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f'Actual Class: {actual_class}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')

    # Add color bar for activation heatmap scale
    plt.colorbar()
    
    # Save the plot
    plt.savefig(filename)
    plt.close()

def plot_grad_cam_for_image(model, image_path, actual_class=None, target_class=None, filename=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # Compute Grad-CAM with the specified or predicted target class
    grad_cam = compute_grad_cam(model, img_tensor, target_class=target_class)
    
    # Get the predicted class for the image
    outputs = model(img_tensor)
    _, predicted_class = torch.max(outputs, 1)

    # Call save_grad_cam_as_image with the original image and classes
    save_grad_cam_as_image(grad_cam, img, predicted_class.item(), actual_class, filename=filename)

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

# Load pre-trained ResNet-101 model
model = resnet152(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one (assuming your dataset has 2 classes)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming 2 output classes

# Move the model to the appropriate device (CPU or GPU)
model = model.to(device)

# Define a loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Training loop with early stopping based on time limit
num_epochs = 50
start_time = time.time()
time_limit = int(os.getenv('JOB_TIME_LIMIT', '7200000'))
safe_margin = 300

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    for images, labels in train_dataloader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    class_acc = accuracy_per_class(conf_matrix)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_dataloader):.4f}, '
        f'Validation Accuracy: {(correct/total)*100:.2f}%, '
        f'Accuracy per Class: {class_acc}, '
        f'Precision: {precision:.4f}, '
        f'Recall: {recall:.4f}, '
        f'F1 Score: {f1:.4f}')
# Save ROC curve and AUC data after the last epoch
    if epoch == num_epochs - 1:
        all_probs = []
        all_true_labels = []
        model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
        plot_roc_curve_and_output_auc(all_true_labels, all_probs)
    if epoch == num_epochs - 1:
        # Choose an image and its corresponding label from the validation set
        sample_image_path51 = r"Brain Tumor Data Set\Brain Tumor Data Set\Healthy\Not Cancer  (51).jpg" #layer 4
        sample_image_path2419 = r"Brain Tumor Data Set\Brain Tumor Data Set\Brain Tumor\Cancer (2419).jpg" #layer 4
        sample_image_path2373 = r"Brain Tumor Data Set\Brain Tumor Data Set\Brain Tumor\Cancer (2373).jpg" #layer 4
        sample_image_path939 = r"Brain Tumor Data Set\Brain Tumor Data Set\Healthy\Not Cancer  (939).jpg" #layer 4
        sample_image_path331 = r"Brain Tumor Data Set\Brain Tumor Data Set\Healthy\Not Cancer  (331).jpg" #layer 4
        sample_image_path370 = r"Brain Tumor Data Set\Brain Tumor Data Set\Brain Tumor\Cancer (370).jpg" #layer 4

        # Compute Grad-CAM for the chosen image with its predicted class
        plot_grad_cam_for_image(model, image_path=sample_image_path51, actual_class=0, target_class=0, filename='NewOutputs/resnet152/grad_cam51.png')
        plot_grad_cam_for_image(model, image_path=sample_image_path2419, actual_class=1, target_class=1, filename='NewOutputs/resnet152/grad_cam2419.png')
        plot_grad_cam_for_image(model, image_path=sample_image_path2373, actual_class=1, target_class=1, filename='NewOutputs/resnet152/grad_cam2373.png')
        plot_grad_cam_for_image(model, image_path=sample_image_path939, actual_class=0, target_class=0, filename='NewOutputs/resnet152/grad_cam939.png')
        plot_grad_cam_for_image(model, image_path=sample_image_path331, actual_class=0, target_class=0, filename='NewOutputs/resnet152/grad_cam331.png')
        plot_grad_cam_for_image(model, image_path=sample_image_path370, actual_class=1, target_class=1, filename='NewOutputs/resnet152/grad_cam370.png')

    # Save Confusion Matrix as Image after the last epoch
    if epoch == num_epochs - 1:
        plot_confusion_matrix(conf_matrix, classes=[str(i) for i in range(2)], filename='NewOutputs/resnet152/confusion_matrix_resnet152_trained.png')

    # Save checkpoint after each epoch
    #save_checkpoint(model, optimizer, epoch, filename=f'checkpoint_epoch_{epoch}.pth.tar')

    # Check for early stopping due to time limit
    epoch_duration = time.time() - epoch_start_time
    time_left = time_limit - (time.time() - start_time)

    if time_left < epoch_duration + safe_margin:
        print(f'Breaking the training loop due to time limit. Time left: {time_left:.2f} seconds.')
        break

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
test_class_acc = accuracy_per_class(test_conf_matrix)

# Calculate precision, recall, and F1 score for test set
test_precision = precision_score(all_true_labels, all_predicted_labels)
test_recall = recall_score(all_true_labels, all_predicted_labels)
test_f1 = f1_score(all_true_labels, all_predicted_labels)

print(f'Test Loss: {test_loss/len(test_dataloader):.4f}, '
    f'Test Accuracy: {(correct/total)*100:.2f}%, '
    f'Accuracy per Class: {test_class_acc}, '
    f'Precision: {test_precision:.4f}, '
    f'Recall: {test_recall:.4f}, '
    f'F1 Score: {test_f1:.4f}')