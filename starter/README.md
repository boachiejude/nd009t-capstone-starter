# Inventory Monitoring at Distribution Centers

## Definition

### Project Overview

Most distribution centers all over the world use robots to move objects from one place to another. These robots use bins which contains multiple objects. Determining the number of objects in each bin can be very valuable in order to check that the process is working as expected.

The main goal of this project is to build a ML model that can count the number of objects in each bin in order to track inventory and check that bins have the appropriate number of items in order to reduce stock mismatches.

### Problem Statement

Based on the background, it can be seen that the problem to be resolved here is related to image classification. A ton of images have been provided by our client (Amazon) and a ML model will be built in order to identify the number of objects in each bin.

### Metrics

For this case, we will use the model accuracy (based on the images which are correctly identified by the model) in order to evaluate how the model is performing.

This metric will be used at the end of each epoch in order to observe how the model is improving its results.

### Algorithms and Techniques

In this project a ML model has been built to identify the number of objects in each image. In order to build this ML model, SageMaker has been used, training a model using a ResNet 18 neural network.

ResNet model is widely used for image classification which is pretrained and can be customized in order to categorize images from different use cases. To adapt this pretrained model to our use case, different training jobs will be launched in AWS SageMaker. In addition, hyperparameters tunning jobs has been launched in order to find the most appropriate combination of hyperparameters for our use case.

### Benchmark

Others have worked on the same dataset. Specifically, we can see two GitHub repos:

1. [Amazon Bin Image Dataset (ABID) Challenge by silverbottlep](https://github.com/silverbottlep/abid_challenge)
2. [Amazon Inventory Reconciliation using AI by Pablo Rodriguez Bertorello, Sravan Sripada, Nutchapol Dendumrongsup](https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN)

As can be seen in the conclusions of both projects, the obtained accuracy is approximately 55%. This is a good starting point for our project, but we will try to improve this accuracy.

## Methodology

### Data preprocessing

The first step to train a model is to download an process the data which will be used as the input. As stated before, we have decided to focus on pictures with 0 to 5 objects. Each picture will be assigned a class according to the following rule:

* Class 1 for pictures without objects
* Class 2 for pictures with 1 object
* Class 3 for pictures with 2 objects
* Class 4 for pictures with 3 objects
* Class 5 for pictures with 4 objects
* Class 6 for pictures with 5 objects


In order to download these pictures, a Python script has been created. The script can be found at the project Jupyter notebook (file `sagemaker.ipynb`). Specifically, the script will iterate over the JSON files from the Amazon dataset and will download the picture if it contains 0 to 5 objects and the number of objects downloaded for the specific class is below 1,000. 

Finally, all these pictures were uploaded to S3, as it is the entry point for data for models being trained on AWS.

### Implementation

As stated before, I planned to use a ResNet neuronal network to train the model. As a base I used [this Python training script](https://github.com/jdboachie/nd009t-capstone-starter/blob/master/starter/train.py)

## Results

### Model Evaluation and Validation

After training the model, the accuracy obtained was 0.39. This is a good starting point, but we can try to improve this accuracy by using hyperparameters tunning.

[Projected cut short due to time limitations]
