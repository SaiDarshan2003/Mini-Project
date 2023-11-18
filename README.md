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
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
```
Loading Datasets:
```
data = []
paths = []
result = []

for r, d, f in os.walk(r'/content/drive/MyDrive/brain_tumor_dataset/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())

paths = []
for r, d, f in os.walk(r"/content/drive/MyDrive/brain_tumor_dataset/no"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())
```
Resizing and transforming:
```
data = np.array(data)
data.shape
result = np.array(result)
result = result.reshape(139,2)
```
Splitting the dataset:
```
x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)
```
Model:
```
model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax')
print(model.summary())
```
Fitting the model:
```
history = model.fit(x_train, y_train, epochs = 28, batch_size = 40, verbose = 1,validation_data = (x_test, y_test))
```
Testing:
```
from matplotlib.pyplot import imshow
img = Image.open(r"/content/yesvt.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence This Is A ' + names(classification))
```
## Output:

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/ccce29a5-237d-451d-a192-0e69d3951da2)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/b5c1a6ab-57ae-4000-9808-a8e1e538cb5d)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/f91069e0-27c9-4f03-858c-b4e7413d4ef5)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/f5571b9f-c7cb-4992-be35-61e91d3b68ca)

![image](https://github.com/SaiDarshan2003/Mini-Project/assets/94692595/a0da1d3d-a2cf-490e-9a8a-a4c013e9464a)


## Result:

In conclusion, the brain tumor detection program using CNN and torchvision demonstrates the potential for accurate and efficient automated diagnosis from MRI scans.
This approach can significantly improve early detection, reduce human error, and streamline the diagnostic process. It represents a promising tool in the field of medical imaging, 
with potential for enhancing patient outcomes and healthcare efficiency.However, it's essential to address ethical considerations, regulatory compliance, and the need for continuous
model improvement for real-world deployment.

