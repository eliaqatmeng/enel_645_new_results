import os
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import torch.nn.functional as F
from torchvision.transforms import Resize
import random

random.seed(42)

# Define the output file path
output_file = 'apr24/simplecnn/simplecnn_output.txt'
roc_curve_png = 'apr24/simplecnn/ROC_curve_simplecnn.png'
roc_curve_txt = 'apr24/simplecnn/ROC_data_simplecnn.csv'
confusion_matrix_png= 'apr24/simplecnn/confusion_matrix_simplecnn.png'

best_f1 = 0.0
best_model = None

num_epochs = 15

# List of image paths
image_paths = [
    r"C:\Users\Gigabyte\Downloads\enel_645_final\slices\Cancer (1413).jpg", 
    r"C:\Users\Gigabyte\Downloads\enel_645_final\slices\Cancer (1609).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\slices\Cancer (2406).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\slices\Cancer (2425).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\slices\Not Cancer  (13).jpg",
    r"C:\Users\Gigabyte\Downloads\enel_645_final\slices\Not Cancer  (52).jpg"]
sys.stdout = open(output_file, 'w')

# Check if CUDA GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def save_checkpoint(model, optimizer, epoch, filename):
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
        
def plot_roc_curve_and_output_auc(labels, probs, filename_png=None, filename_csv=None):
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

# Define the root directory containing the "brain tumor" and "healthy" folders
root_dir = r"C:\Users\Gigabyte\Downloads\Brain Tumor Data Set\Brain Tumor Data Set"

# Transform dataset input to match inputs for SimpleCNN model
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
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

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):  # Specify the number of output classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Change 32 to 16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust input size based on your image dimensions
        self.fc2 = nn.Linear(512, num_classes)  # Adjust output size based on the number of classes

        # Register hook for the last convolutional layer
        self.conv3.register_forward_hook(self.hook_fn)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Make sure the hook is registered here
        # Change 32 to 16
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def hook_fn(self, module, input, output):
        self.last_conv_output = output

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
    # Adjust this according to your model architecture
    model.conv3.register_forward_hook(hook)
    
    model(img_tensor)
    heatmap = torch.mean(last_conv_layer, dim=1)
    for i in range(heatmap.size(0)):
        heatmap[i, :, :] *= pooled_gradients[i]

    heatmap = F.relu(torch.sum(heatmap, dim=0)).detach().cpu().numpy()

    return heatmap

# Initialize your simple CNN model
num_classes = 2  # Assuming 2 output classes
model = SimpleCNN(num_classes=num_classes)

# Move the model to the appropriate device
model = model.to(device)

# Define a loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


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


    if f1 > best_f1:
        best_f1 = f1
        best_model = model.state_dict()
        print("\nBest model updated. F1 score:", best_f1)

    print("Epoch [{}/{}], F1 Score: {:.4f}".format(epoch+1, num_epochs, f1))

    

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
        plot_roc_curve_and_output_auc(all_true_labels, all_probs, filename_png=roc_curve_png, filename_csv=roc_curve_txt)

    # Save Confusion Matrix as Image after the last epoch
    if epoch == num_epochs - 1:
        plot_confusion_matrix(conf_matrix, classes=[str(i) for i in range(2)], filename=confusion_matrix_png)

    # Save checkpoint after each epoch
    #save_checkpoint(model, optimizer, epoch, filename=f'checkpoint_epoch_{epoch}.pth.tar')

    # Check for early stopping due to time limit
    epoch_duration = time.time() - epoch_start_time
    time_left = time_limit - (time.time() - start_time)

    if time_left < epoch_duration + safe_margin:
        print(f'Breaking the training loop due to time limit. Time left: {time_left:.2f} seconds.')
        break

# Testing
if best_model is not None:
    model.load_state_dict(best_model)
    save_checkpoint(model, optimizer, epoch, 'apr24/simplecnn/checkpoint.pth.tar')

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

print(f'\nTest Loss: {test_loss/len(test_dataloader):.4f}, '
    f'Test Accuracy: {(correct/total)*100:.2f}%, '
    f'Accuracy per Class: {test_class_acc}, '
    f'Precision: {test_precision:.4f}, '
    f'Recall: {test_recall:.4f}, '
    f'F1 Score: {test_f1:.4f}')

# load and preprocess an image
def load_and_preprocess_image(image_path, resize_dim=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, resize_dim)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return image_tensor, image

for image_path in image_paths:
    # Load and preprocess the image
    image_tensor, original_image = load_and_preprocess_image(image_path)

    # Get the predicted class label
    with torch.no_grad():
        model.eval()
        outputs = model(image_tensor)
        _, predicted_class_idx = torch.max(outputs, 1)

    # Map the predicted class index to the actual class label
    class_labels = ['Healthy', 'Brain Tumor'] 
    predicted_class_label = class_labels[predicted_class_idx]

    # Print the predicted class label
    print(f"\nPredicted Class Label: {predicted_class_label}")

    # Get the Grad-CAM heatmap
    heatmap = compute_grad_cam(model, image_tensor)

    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    heatmap = np.uint8(255 * heatmap / np.max(heatmap))

    # Apply colormap to the heatmap
    heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = heatmap_colormap * 0.5 + original_image * 0.5
    superimposed_img = np.uint8(superimposed_img)

    # Convert the superimposed image to RGB (cv2 reads images in BGR format)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    os.makedirs("apr24/simplecnn", exist_ok=True)

    output_image_path = f"apr24/simplecnn/{os.path.basename(image_path).split('.')[0]}_gradcam_output.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM image saved as {output_image_path}")
