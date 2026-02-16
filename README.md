  
  # Blood Cell Type Image Classification using Transfer Learning (MobilenetV2)


<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-11557c.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-0080bf.svg?logo=matplotlib&logoColor=white">
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-00acdf.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-55d0ff.svg?logo=TensorFlow&logoColor=white"></a>
<a href="#"><img alt="Keras" src="https://img.shields.io/badge/Keras-7ce8ff.svg?logo=Keras&logoColor=white"></a>
<!--<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-00acdf.svg?logo=pandas&logoColor=white"></a>-->
<br />
Blood Cell Type Prediction
<br />
<!-- by **[Pratik Sahu](https://www.linkedin.com/in/abcd123/)** -->
Apr 2023

## Project Description
This project aims to classify blood cell types using deep learning and transfer learning techniques. The MobileNetV2 architecture is employed as a pre-trained model for feature extraction, and a custom classification model is built on top of it.
  
## Project Goal

The goal of this project is to develop an automated method for detecting and classifying blood cell subtypes using machine learning techniques. The project utilizes a dataset containing 12,500 augmented images of blood cells, each labeled with the corresponding cell type. The dataset consists of four different cell types: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. 
In this project we will build a **deep learning** based image classification model using a pretrained model MobilenetV2 and applying the concept of **transfer learning** 
  
## Transfer Learning

### What is Transfer Learning?
Transfer learning is a technique that uses knowledge gained from a pre-trained model on one task to boost performance on a different but related task. It takes advantage of the learned features and representations from the pre-trained model, which can be applied to a new task. This approach helps overcome data limitations, speeds up training, and improves the performance of models on new tasks by leveraging the knowledge encoded in the pre-trained model.

### Benefits of Transfer Learning

Transfer learning offers several benefits:

1. **Improved Performance**: By leveraging pre-trained models that have been trained on large-scale datasets, transfer learning can help achieve higher performance on new tasks with limited available data.

2. **Faster Training**: Transfer learning reduces the training time and computational resources required to build models from scratch, as the pre-trained model has already learned generic features that can be used as a starting point.

3. **Generalization**: Transfer learning allows models to generalize better to new tasks by learning abstract and generic representations from a diverse range of data. This helps in capturing higher-level features that can be relevant across different tasks.

4. **Data Efficiency**: With transfer learning, models can achieve good performance even with smaller datasets, as they can leverage the knowledge learned from large-scale datasets.

Overall, transfer learning is a powerful technique that enables the transfer of knowledge and representations learned from one task to another, leading to improved performance, faster training, and better generalization capabilities.
  
<!--## Setup for Python

1. Install Python ([Setup instructions](https://wiki.python.org/moin/BeginnersGuide))

2. Install dependent packages

```
pip3 install -r training/requirements.txt
```
  -->


## Project Overview

The project consists of the following key steps:
#### :one:   Data Acquisition
  You can download the data from [kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells).

#### :two:   Data Preprocessing
  The image data is loaded and preprocessed using the `ImageDataGenerator` class from TensorFlow. Data is split into training and validation sets.

#### :three: Build Pretrained Model: 
  The MobileNetV2 model is loaded with pre-trained weights and frozen to retain its learned features. Additional layers are added to the model for classification purposes.

#### :four: Training the Model: 
  The model is trained using the training images and validated using the validation images. Training is performed for a specified number of epochs, and early stopping is applied to prevent overfitting.

#### :five: Evaluating Results: 
  The trained model is evaluated on the test data. Metrics such as loss, accuracy, and a classification report are computed. Additionally, a confusion matrix is plotted to analyze the model's performance.

 ## Getting Started

To reproduce this project, follow these steps:

1. Dataset: Prepare a dataset of blood cell images, divided into training and testing directories. The `train_dir` and `test_dir` variables in the code should be updated with the respective directory paths.

2. Environment Setup: Set up a Python environment with TensorFlow and the necessary dependencies. Use `pip` or `conda` to install the required packages. 

3. Code Execution: Execute the code cells in the provided Jupyter Notebook [model](./model.ipynb) in sequential order.

4. Analyze Results: Review the training and validation loss/accuracy plots to assess the model's performance. Evaluate the model on the test data and examine the confusion matrix and classification report.

## Results

The project yields the following conclusions based on the validation set:

- The model achieves a high accuracy of 94.5% on the validation images.
- There are two possible explanations for the subpar performance on the test set:
  1. The test data may contain mislabeled samples.
  2. The test set might have been generated differently from the original training set, leading to inconsistencies in image quality or class representation.

The confusion matrix provides additional insights, suggesting that the neutrophil class in the test set could contain mixed samples from other classes.

## Conclusion

This project demonstrates the application of transfer learning using the MobileNetV2 architecture for blood cell type classification. Despite achieving high accuracy on the validation set, further investigation is required to understand the discrepancies in performance on the test set. The project can be extended and optimized by implementing various strategies to improve the model's accuracy and generalization capability.
  
## Further Improvements

To enhance the project, consider the following steps:

1. Data Augmentation: Apply data augmentation techniques to increase the diversity of the training data and improve the model's generalization.

2. Hyperparameter Tuning: Explore different hyperparameter configurations, such as learning rate, batch size, and number of epochs, to optimize the model's performance.

3. Model Selection: Experiment with other pre-trained models, such as Xception or ResNet, to compare their performance with MobileNetV2.

4. Ensemble Methods: Combine multiple models or use ensemble techniques to enhance classification accuracy and robustness.

5. Fine-tuning: Gradually unfreeze and fine-tune the pre-trained layers of the model to adapt to the specific blood cell classification task.
