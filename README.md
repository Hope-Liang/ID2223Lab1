# ID2223Lab1
Authors: (GROUP 31) Khalid El Yaacoub, Xinyu Liang

## Introduction
In this project, the goal was to build serverless ML systems using Hopsworks as a feature and model store, as well as using Hugging face as an interactive UI for our applications for the Iris Flowers dataset as well as a Titanic survival dataset. Both datasets are a classification problem where in the Iris flowers dataset the goal is to predict the type of flower based on some flower features while in the Titanic dataset the goal is to predict whether a passenger will survive (binary). Two UIs were created for each dataset. One interactive UI where users can input features values to obtain a prediction, and another dashboard UI showing the most recent prediction and the actual label for that prediction as well as a confusion matrix showing the historical model performance.

## Iris Flowers Dataset
### Feature & Model store on Hopsworks
The first step was to run the [iris-feature-pipipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-feature-pipeline.py) file, which stores the Iris dataset on the Hopsworks platform by creating a feature group. Now that the features have been stored on Hopsworks, the next step is to train a Machine Learning model and store the model on Hopsworks. As such, we ran the [iris-training-pipeline.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/iris-training-pipeline.py) file which creatues a feature view from the Iris feature group and then trains a KNN classifier. The KNN classifier model is then uploaded into the Hopsworks Model Registry and is now ready to be served into an interactive UI. 

### Interactive UI - Iris
To create the interactive UI on Hugging Face, we ran the [app.py](https://github.com/Hope-Liang/ID2223Lab1/blob/main/serverless-ml-iris/huggingface-spaces-iris/app.py). This application file loads the KNN model from Hopsworks and allows users to input certain flower features, using gradio interface, to obtain a prediction on what type of flower this would be. An image of that flower type is then outputted to the user.


## Links to Apps

The links for the four pages: <br/>
1. [Iris Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris) <br/>
2. [Iris Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris-monitor) <br/>
3. [Titanic Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic) <br/>
4. [Titanic Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic-monitor) <br/>


