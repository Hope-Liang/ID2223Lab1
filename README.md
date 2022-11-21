# ID2223Lab1
Authors: (GROUP 31) Khalid El Yaacoub, Xinyu Liang

## Introduction
In this project, the goal was to build serverless ML systems using Hopsworks as a feature and model store, as well as using Hugging face as an interactive UI for our applications for the Iris Flowers dataset as well as a Titanic survival dataset. Both datasets are a classification problem where in the Iris flowers dataset the goal is to predict the type of flower based on some flower features while in the Titanic dataset the goal is to predict whether a passenger will survive (binary). Two UIs were created for each dataset. One interactive UI where users can input features values to obtain a prediction, and another dashboard UI showing the most recent prediction and the actual label for that prediction as well as a confusion matrix showing the historical model performance.

## Iris Flowers Dataset
### Feature & Model store on Hopsworks
The first step was to run the [iris-feature-pipipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-feature-pipeline.py) file, which stores the Iris dataset on the Hopsworks platform by creating a feature group. Now that the features have been stored on Hopsworks, the next step is to train a Machine Learning model and store the model on Hopsworks. As such, we ran the [iris-training-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-training-pipeline.py) file which creatues a feature view from the Iris feature group and then trains a KNN classifier. The KNN classifier model is then uploaded into the Hopsworks Model Registry and is now ready to be served into an interactive/dashboard UI. 

### Interactive UI - Iris
To create the interactive UI on Hugging Face, we ran the [app.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/huggingface-spaces-iris/app.py) file. This application file loads the KNN model from Hopsworks and allows users to input certain flower features, using gradio interface, to obtain a prediction on what type of flower this would be. An image of that flower type is then outputted to the user.

### Dashboard UI - Iris
To create the dashboard UI, we firstly ran the [iris-batch-inference-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-batch-inference-pipeline.py) file. This file loads the model from the Hopsworks Model Registry and then performs batch inference on data stored in the feature view on Hopsworks. It then uploads an image of the predicted flower type and an image of the actual flower type into a 'Resources/images' folder on Hopsworks as well as the 5 most recent predictions. This file is run multiple times until there are 3 different flower predictions to create the confusion matrix before finally uploading a confusion matrix to the same resources file outlying the historical model performance. Finally, the [app.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/huggingface-spaces-iris-monitor/app.py) file creates the dashboard UI on Hugging Face by showing all the images stored in the 'Resources/images' folder on Hopsworks, which includes the 5 most recent prediction history as well as the Confusion Maxtrix with Historical Prediction Performance.


## Titanic survival dataset

### Preprocessing
Before storing the dataset on Hopsworks, we first had to preprocess the data. To this end, we firstly dropped some columns, such as 'name', that we deemed irrelevant to the prediction task. We then created a preprocessing pipeline that performs onehot encoding on nominal categorical features and then imputes any missing values with the mode (most frequent values). The details are shown in the [preprocessor_pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/preprocessor_pipeline.py) file. The titanic dataset is now processed and ready for storage on Hopsworks.

### Feature & Model store on Hopsworks
The steps are very similar to the Iris dataset except for some slight modifications. We ran the [titanic-feature-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/titanic-feature-pipeline.py) file which simply stores the processed titanic dataset in a feature group on Hopsworks. The next step is to train a Machine Learning model, XGBoost in our case, and store the model on Hopsworks. We ran the [titanic-training-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/titanic-training-pipeline.py) file which creatues a feature view from the titanic feature group and then trains an XGBoost classifier. The model is then uploaded into the Hopsworks Model Registry and is now ready to be served into an interactive/dashboard UI.

### Interactive UI - Titanic
To create the interactive UI on Hugging Face, we ran the [app.py](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic/blob/main/app.py) file. This application file loads the XGBoost model from Hopsworks and allows users to input certain passenger features, using gradio interface, to obtain a prediction on whether said passenger would survive or not. An image of Rose or Jack (from titanic) is shown based on whether the passenger survives or not.

### Dashboard UI - Titanic
To create the dashboard UI, we firstly ran the [titanic-batch-inference-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/titanic-batch-inference-pipeline.py) file. This file loads the model from the Hopsworks Model Registry and then performs batch inference on data stored in the feature view on Hopsworks. It then uploads an image of Rose or Jack (survived or not survived) and an image of the actual outcome into a 'Resources/images-titanic' folder on Hopsworks as well as the 5 most recent predictions. This file is run multiple times until there are 2 different passenger outcome predictions to create the confusion matrix before finally uploading a confusion matrix to the same resources file outlying the historical model performance. Finally, the [app.py](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic-monitor/blob/main/app.py) file creates the dashboard UI on Hugging Face by showing all the images stored in the 'Resources/images' folder on Hopsworks, which includes the 5 most recent prediction history as well as the Confusion Maxtrix with Historical Prediction Performance.

## Links to Apps

The links for the four pages: <br/>
1. [Iris Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris) <br/>
2. [Iris Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris-monitor) <br/>
3. [Titanic Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic) <br/>
4. [Titanic Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic-monitor) <br/>


