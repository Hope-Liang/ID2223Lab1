# ID2223Lab1
Authors: (GROUP 31) Khalid El Yaacoub, Xinyu Liang

## Introduction
In this project, the goal was to build serverless ML systems using Hopsworks as a feature and model store, as well as using Hugging face as an interactive UI for our applications for the Iris Flowers dataset as well as a Titanic survival dataset. Both datasets are a classification problem where in the Iris flowers dataset the goal is to predict the type of flower based on some flower features while in the Titanic dataset the goal is to predict whether a passenger will survive (binary).

## Iris Flowers Dataset
The first step was to run the 'iris-feature-pipipeline.py' file, which stores the Iris dataset on the Hopsworks platform by creating a feature group. Now that the features have been stored on Hopsworks, the next step is to train a Machine Learning model and store the model on Hopsworks. As such, we ran the 'iris-training-pip
ipeline.py' file which creatues a feature view from the Iris feature group and then trains a KNN classifier. The KNN classifier model is then uploaded into the Hopsworks Model Registry and is now ready to be served into an interactive UI. To create the UI on Hugging Face, we ran the 'app.py' in the 'serverless-ml-iris/huggingface-spaces-iris' folder.


## Links to Apps

The links for the four pages: <br/>
1. [Iris Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris) <br/>
2. [Iris Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-iris-monitor) <br/>
3. [Titanic Interactive](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic) <br/>
4. [Titanic Dashboard](https://huggingface.co/spaces/HopeLiang/huggingface-spaces-titanic-monitor) <br/>


