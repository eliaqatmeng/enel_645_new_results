import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_multiple_roc_curves(csv_files, labels, filename_png=None):
    plt.figure()
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        fpr = df['False Positive Rate']
        tpr = df['True Positive Rate']
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if filename_png:
        plt.savefig(filename_png)
    else:
        plt.show()

csv_files_1 = ["apr24/densenet/ROC_data_densenet_trained.csv", 
             "apr24/resnet50/ROC_data_resnet50_trained.csv",
             "apr24/resnet101/ROC_data_resnet101_trained.csv",
             "apr24/resnet152/ROC_data_resnet152_trained.csv",
             "apr24/vgg16/ROC_data_VGG16_trained.csv"]
labels_1 = ['Densenet', 'Resnet50', 'Resnet101', 'Resnet152', 'VGG16']
plot_multiple_roc_curves(csv_files_1, labels_1, filename_png='multiple_roc_curves_trained.png')

csv_files_2 = ["apr24/densenet_untrained/ROC_data_densenet_untrained.csv", 
             "apr24/resnet50_untrained/ROC_data_resnet50_untrained.csv",
             "apr24/resnet101_untrained/ROC_data_resnet101_untrained.csv",
             "apr24/resnet152_untrained/ROC_data_resnet152_untrained.csv",
             "apr24/vgg16_untrained/ROC_data_VGG16_untrained.csv",
             "apr24/simplecnn/ROC_data_simplecnn.csv"]
labels_2 = ['Densenet', 'Resnet50', 'Resnet101', 'Resnet152', 'VGG16', 'SimpleCNN']
plot_multiple_roc_curves(csv_files_2, labels_2, filename_png='multiple_roc_curves_untrained.png')
