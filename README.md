# Brain Tumour Detection

Brain tumor detection is a crucial aspect of medical diagnosis and treatment planning. 
The human brain is a complex organ, and the presence of a tumor can have significant implications for an individual's health. 
Early detection plays a key role in improving the chances of successful treatment and better outcomes.

## Features

### Image Preprocessing:

Normalization: Ensures consistent pixel values for improved model performance.
Resizing: Standardizes image dimensions for compatibility with the detection model.
Augmentation: Enhances the dataset by applying transformations like rotations or flips.

### Convolutional Neural Network (CNN):

Utilizes convolutional layers to automatically learn hierarchical features from input images.
Employs pooling layers for downsampling and reducing spatial dimensions.
Utilizes fully connected layers for making predictions based on learned features.

### Transfer Learning:

Leverages pre-trained CNN models (e.g., ResNet, VGG) for feature extraction.
Fine-tunes the model on a brain tumor detection-specific dataset.

### Data Splitting:

Segregates the dataset into training, validation, and test sets for model development and evaluation.

### Loss Function and Optimization:

Employs appropriate loss functions (e.g., binary cross-entropy) to measure model performance.
Optimizes model parameters using optimization algorithms (e.g., SGD, Adam).

### Metrics for Evaluation:

Utilizes metrics such as accuracy, precision, recall, F1-score, and ROC-AUC for model evaluation.

### Post-Processing Techniques:

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
from os import listdir
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from PIL import Image,ImageEnhance
import random
import warnings
warnings.filterwarnings("ignore")
```
Loading Datasets:
```
os.makedirs("/kaggle/working/data-augmentation")
os.makedirs('/kaggle/working/data-augmentation/yes')
os.makedirs('/kaggle/working/data-augmentation/no')
input_dir="/content/drive/MyDrive/brain_tumor_dataset/"
output_dir="/kaggle/working/data-augmentation/"
```
Data Augmentation:
```
def data_augmentation(input_dir,output_dir,image_num_no=5,image_num_yes=8):

    for files in listdir(input_dir):
        for file in listdir(input_dir+files):
            if(file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")):
                input_path=os.path.join(input_dir+files,file)
                image=Image.open(input_path).convert("L")

                if(files=="no"):
                    for i in range(image_num_no):
                        augmented_image=apply_augmentation(image)
                        output_file=f"{os.path.splitext(file)[0]}_{i}.png"
                        output_path=os.path.join(output_dir+files,output_file)
                        augmented_image.save(output_path)

                elif(files=="yes"):
                    for i in range(image_num_yes):
                        augmented_image = apply_augmentation(image)
                        output_file = f"{os.path.splitext(file)[0]}_{i}.png"
                        output_path = os.path.join(output_dir+files, output_file)
                        augmented_image.save(output_path)
def apply_augmentation(image):

    angle=random.randint(-20,20)
    augmented_image=image.rotate(angle)

    if(random.random()>0.5):
        augmented_image=augmented_image.transpose(Image.FLIP_LEFT_RIGHT)

    brightness_factor=random.uniform(0.85,1.15)
    enhanced_image= ImageEnhance.Brightness(augmented_image).enhance(brightness_factor)

    contrast_factor = random.uniform(0.85, 1.15)
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast_factor)

    color_image = ImageEnhance.Color(enhanced_image).enhance(random.uniform(0.8, 1.2))

    # nearest-neighbor

    scale_factor=random.uniform(0.8,1.2)
    new_size=(int(color_image.width * scale_factor),int(color_image.height * scale_factor))

    new_image=color_image.resize(new_size,Image.NEAREST)


    return new_image
data_augmentation(input_dir=input_dir,output_dir=output_dir,image_num_no=5,image_num_yes=5)
```
Dataset:
```
class CustomDataset(Dataset):

    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir
        self.transform=transform

        self.images=[]
        self.labels=[]

        for subdir,_,files in os.walk(root_dir):
            for file in files:
                if(file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")):
                    self.images.append(os.path.join(subdir,file))

                    label=1 if "yes" in subdir else 0
                    self.labels.append(label)


    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image_path=self.images[idx]
        image=Image.open(image_path).convert("L")

        if(self.transform):
            image=self.transform(image)

        label=self.labels[idx]

        return image,label
```
Transforms:
```
transform=transforms.Compose([transforms.Resize((240,240)),
                              transforms.ToTensor(),
                               transforms.Normalize([0.485],[0.229])
                              ])

root_dir="/kaggle/working/data-augmentation/"
train_test_dataset=CustomDataset(root_dir=root_dir,transform=transform)
```
Train Test Split:
```
train_size=0.7
test_size=0.15
valid_size=1-(train_size+test_size)

total_data=len(train_test_dataset)
indices=list(range(total_data))

train_indices, test_indices = train_test_split(indices, test_size=(test_size + valid_size),random_state=42)
test_indices, valid_indices = train_test_split(test_indices, test_size=(valid_size / (test_size + valid_size)),random_state=42)

print(" Number of train indices:{}\n Number of test indices:{}\n Number of validation indices:{}".format(len(train_indices),len(test_indices),len(valid_indices)))

train_dataset=torch.utils.data.Subset(train_test_dataset,train_indices)
test_dataset=torch.utils.data.Subset(train_test_dataset,test_indices)
valid_dataset=torch.utils.data.Subset(train_test_dataset,valid_indices)

batch_size = 64

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size)
valid_loader=DataLoader(valid_dataset,batch_size=batch_size)
```
VGG-16 Function:
```
class VGG16(nn.Module):

    def __init__(self,num_classes=1000):
        super(VGG16,self).__init__()

        self.features=nn.Sequential(

            nn.Conv2d(1,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),


            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),


            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),


            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )


        self.avgpool=nn.AdaptiveAvgPool2d((7,7))
        self.classifier=nn.Sequential(nn.Linear(512*7*7,4096),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Linear(4096,4096),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(4096,num_classes)
                                     )

        self._initialize_weights()


    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.classifier(x)

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
```
Train Function:
```
def train(model,train_loader,optimizer,criteron):

    model.train()
    train_loss=0
    correct=0
    total=0

    for images,labels in train_loader:
        images,labels =images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _,predicted=outputs.max(1)
        total+=labels.size(0)
        correct+=predicted.eq(labels).sum().item()

    train_accuracy=100*correct/total
    train_loss/=len(train_loader)
    return train_loss,train_accuracy
```
Validation Function:
```
def validate(model,valid_loader,criterion):
    model.eval()
    val_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in valid_loader:
            images,labels=images.to(device),labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            val_loss+=loss.item()
            _,predicted=outputs.max(1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()

        val_accuracy = 100.0 * correct / total
        val_loss /= len(valid_loader)
    return val_loss, val_accuracy
```
Training:
```
train_accuracy=[]
validation_accuracy=[]
train_losses=[]
validation_losses=[]

for epoch in range(epochs):
    train_loss, train_acc = train(vgg16_kaiming, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(vgg16_kaiming, valid_loader, criterion)

    train_accuracy.append(train_acc)
    validation_accuracy.append(val_acc)
    train_losses.append(train_loss)
    validation_losses.append(val_loss)


    print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")
```
## Output:

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/a101f1f6-b042-4b62-aecd-17ce50e99aa2)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/8d518eb7-238f-4f01-b5c8-e22077614ce6)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/e6270987-8725-4585-a620-39e46499a4c7)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/7c4db580-b650-4516-ad91-6d645b728fd7)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/bee42948-7e75-4b52-8471-02a0637b311a)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/1ed0cbd7-1d6a-4e57-8a86-634cff42df34)


## Result:

In conclusion, the brain tumor detection program using CNN and torchvision demonstrates the potential for accurate and efficient automated diagnosis from MRI scans.
This approach can significantly improve early detection, reduce human error, and streamline the diagnostic process. It represents a promising tool in the field of medical imaging, 
with potential for enhancing patient outcomes and healthcare efficiency.However, it's essential to address ethical considerations, regulatory compliance, and the need for continuous
model improvement for real-world deployment.

