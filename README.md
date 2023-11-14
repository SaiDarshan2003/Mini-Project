# Brain Tumour Detection

Brain tumor detection is a crucial aspect of medical diagnosis and treatment planning. 
The human brain is a complex organ, and the presence of a tumor can have significant implications for an individual's health. 
Early detection plays a key role in improving the chances of successful treatment and better outcomes.

## Features

Image Preprocessing:
Normalization: Ensures consistent pixel values for improved model performance.
Resizing: Standardizes image dimensions for compatibility with the detection model.
Augmentation: Enhances the dataset by applying transformations like rotations or flips.

Convolutional Neural Network (CNN):
Utilizes convolutional layers to automatically learn hierarchical features from input images.
Employs pooling layers for downsampling and reducing spatial dimensions.
Utilizes fully connected layers for making predictions based on learned features.

Transfer Learning:
Leverages pre-trained CNN models (e.g., ResNet, VGG) for feature extraction.
Fine-tunes the model on a brain tumor detection-specific dataset.

Data Splitting:
Segregates the dataset into training, validation, and test sets for model development and evaluation.

Loss Function and Optimization:
Employs appropriate loss functions (e.g., binary cross-entropy) to measure model performance.
Optimizes model parameters using optimization algorithms (e.g., SGD, Adam).

Metrics for Evaluation:
Utilizes metrics such as accuracy, precision, recall, F1-score, and ROC-AUC for model evaluation.

Post-Processing Techniques:
Applies post-processing methods to refine model predictions, such as thresholding or morphological operations.

## Requirements

- Python 3.x
- Required Python packages: numoy,pandas,os,opendatasets,torchvision,transforms,datasets,torch

## Architecture Diagram/Flow

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/fcaa1a7a-92dd-43da-ba9b-90559ef5c685)


## Usage

1. Open a Python file.

2. Import the necessary packages.

3. Load the dataset.

4. Split the dataset into training and testing images.

5. Create a neural classififcation function to classify the images.

6. Create an evaluation function that corresponds with the neural network classification function.

7. Compile and fit the neural model with appropriate optimizer,loss functions and also the epochs,batch_sizes etc respectively.

8. Prepare the loop for loading the testing images.

## Program:

Packages Used:
```
import numpy as np
import pandas as pd
import os
import opendatasets as od
import pandas
from google.colab import files
import zipfile
from torchvision.transforms import ToTensor, Compose,Resize
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
```
Loading Datasets:
```
data_paths='/content/drive/MyDrive/brain_tumor_dataset'
```
Resizing and transforming:
```
transfm=Compose([ToTensor(),Resize((224,224))])
dataset=ImageFolder(data_paths, transform=transfm)
```
Splitting the dataset:
```
train_set,valid_set=torch.utils.data.random_split(dataset,[200,53])
train_loader=DataLoader(train_set, batch_size=4, shuffle=True)
valid_loader=DataLoader(valid_set, batch_size=4)
```
Model:
```
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,256,(3,3)),
            nn.ReLU(),
            nn.Conv2d(256,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(224-6)*(224-6), 64),
            nn.Linear(64,64),
            nn.Linear(64,2)
            )
    def forward(self,x):
        return self.model(x)
```
Optimizer and fitting the model:
```
clf=Classifier()
opt=Adam(clf.parameters(),lr=1e-5)
loss_fn=nn.CrossEntropyLoss()
```
Loading the test dataset:
```
    for X,y in train_loader:
        X,y=X,y
        yhat=clf(X)
        loss=loss_fn(yhat,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred=torch.argmax(yhat,1)
        correct+=(y==pred).sum().item()
        items+=y.size(0)
    print(f"Epoch {epoch} loss {loss} train_acc {correct*100/items}")
```
## Output:

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/ea961da9-9e22-4066-9e6f-b08df895fd5f)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/85983d88-2d2a-4a6e-9eac-39f62c98e4f4)


## Result:

In conclusion, the brain tumor detection program using CNN and torchvision demonstrates the potential for accurate and efficient automated diagnosis from MRI scans.
This approach can significantly improve early detection, reduce human error, and streamline the diagnostic process. It represents a promising tool in the field of medical imaging, 
with potential for enhancing patient outcomes and healthcare efficiency.However, it's essential to address ethical considerations, regulatory compliance, and the need for continuous
model improvement for real-world deployment.

