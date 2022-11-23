# ID2223Lab1
Authors: (GROUP 31) Khalid El Yaacoub, Xinyu Liang

## Introduction
In this project, the goal was to build serverless ML systems using a pipeline structure: Hopsworks as a feature and model store, Hugging face as an interactive UI to build applications for the Iris-Flowers dataset as well as a Titanic-survival dataset. Modal could be optionally used to schedule the data generation, model training and predictions, while it's not used in our case. Both datasets are used for classification tasks, where in the Iris flowers dataset the goal is to predict the type of flower based on some flower features, while in the Titanic dataset the goal is to predict whether a passenger will survive or not based on his (binary). <br/>
Two UIs were created for each dataset. One interactive UI where users can input feature values to obtain a prediction, and another dashboard UI showing the most recent predictions results as well as a confusion matrix showing the historical model performance.

## Iris Flowers Dataset
The code for this part is given by Professor Jim Dowling from his [github repo](https://github.com/ID2223KTH/id2223kth.github.io).

### Feature & Model store on Hopsworks
The first step was to run the [serverless-ml-iris/iris-feature-pipipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-feature-pipeline.py) file, which stores the Iris dataset on the Hopsworks platform by creating a feature group. Now that the features have been stored on Hopsworks, the next step is to train a Machine Learning model and store the model on Hopsworks. As such, we ran the [serverless-ml-iris/iris-training-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-training-pipeline.py) file which creatues a feature view from the Iris feature group and then trains a KNN classifier. The KNN classifier model is then uploaded into the Hopsworks Model Registry and is now ready to be served into an interactive/dashboard UI. 

### Interactive UI - Iris
To create the interactive UI on Hugging Face, we ran the [serverless-ml-iris/huggingface-spaces-iris/app.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/huggingface-spaces-iris/app.py) file. This application file loads the KNN model from Hopsworks and allows users to input certain flower features, using gradio interface, to obtain a prediction on what type of flower it is most likely to be. An image of that flower type is then shown to the user.

### Dashboard UI - Iris
To create the dashboard UI, we firstly ran the [serverless-ml-iris/iris-feature-pipeline-daily.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-feature-pipeline-daily.py) to generate new synthetic data points and use the [serverless-ml-iris/iris-batch-inference-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-batch-inference-pipeline.py) file to make predictions on it. This is done by loading the model from the Hopsworks Model Registry and then performs batch inference on data stored in the feature view on Hopsworks. It then uploads an image of the predicted flower type and an image of the actual flower type into a 'Resources/images' folder on Hopsworks as well as the 5 most recent predictions. This file is run multiple times until there are 3 different flower predictions to create the confusion matrix before finally uploading a confusion matrix to the same place visualizing the historical model performance. <br/>
Finally, the [serverless-ml-iris/huggingface-spaces-iris-monitor/app.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/huggingface-spaces-iris-monitor/app.py) file creates the dashboard UI on Hugging Face by showing all the images stored in the 'Resources/images' folder on Hopsworks, which includes the truth and prediction images for the daily-generated new features, and the 5 most recent prediction history as well as the Confusion Maxtrix with historical prediction performance.


## Titanic Survival dataset
The code for this part is modified from the one for Iris Flower.

### Preprocessing
Before storing the dataset on Hopsworks, we first had to preprocess the data. To this end, we firstly dropped some columns, such as 'name', that we deemed irrelevant to the prediction task. We then created a preprocessing pipeline that performs onehot encoding on nominal categorical features and then imputes any missing values with the mode (most frequent values). The details are shown in the [serverless-ml-titanic/preprocessor_pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/preprocessor_pipeline.py) file. The titanic dataset is then processed and ready for storage on Hopsworks.

### Feature & Model store on Hopsworks
The steps are very similar to the Iris dataset except for some slight modifications. We ran the [serverless-ml-titanic/titanic-feature-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/titanic-feature-pipeline.py) file which simply stores the processed titanic dataset in a feature group on Hopsworks. The next step is to train a Machine Learning model, XGBoost in our case, and store the model on Hopsworks. We ran the [serverless-ml-titanic/titanic-training-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/titanic-training-pipeline.py) file which creates a feature view from the titanic feature group and then trains an XGBoost classifier. The model is then uploaded into the Hopsworks Model Registry and is now ready to be served into an interactive/dashboard UI.

### Interactive UI - Titanic
To create the interactive UI on Hugging Face, we ran the [serverless-ml-titanic/huggingface-spaces-titanic/app.py](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic/blob/main/app.py) file. This application file loads the XGBoost model from Hopsworks and allows users to input certain passenger features, using gradio interface, to obtain a prediction on whether said passenger would survive or not. An image of Rose or Jack as its label (from [titanic movie](https://en.wikipedia.org/wiki/Titanic_(1997_film))) is shown based on whether the passenger survives or not.

### Dashboard UI - Titanic
To create the dashboard UI, we firstly ran the [titanic-batch-inference-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-titanic/titanic-batch-inference-pipeline.py) file. This file loads the model from the Hopsworks Model Registry and then performs batch inference on data stored in the feature view on Hopsworks. It then uploads an image of Rose or Jack (survived or not survived) and an image of the actual outcome into a 'Resources/images-titanic' folder on Hopsworks as well as the 5 most recent predictions. This file is run multiple times until there are 2 different passenger outcome predictions to create the confusion matrix before finally uploading a confusion matrix to the same resources file outlying the historical model performance. Finally, the [app.py](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic-monitor/blob/main/app.py) file creates the dashboard UI on Hugging Face by showing all the images stored in the 'Resources/images' folder on Hopsworks, which includes the 5 most recent prediction history as well as the Confusion Maxtrix with Historical Prediction Performance.

## Links to Apps

The links for the four pages: <br/>
1. [Iris Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris) <br/>
2. [Iris Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris-monitor) <br/>
3. [Titanic Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic) <br/>
4. [Titanic Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic-monitor) <br/>


