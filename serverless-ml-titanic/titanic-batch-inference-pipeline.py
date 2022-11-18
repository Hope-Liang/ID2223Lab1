import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    import xgboost as xgb
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    # print(y_pred)
    survival = y_pred[y_pred.size-1]
    if survival == 1.0:
        survival_url = "https://raw.githubusercontent.com/Hope-Liang/ID2223Lab1/main/serverless-ml-titanic/images/survived.png"
        print("Survival predicted: Survived.")
        survival_str = "survived"
    else:
        survival_url = "https://raw.githubusercontent.com/Hope-Liang/ID2223Lab1/main/serverless-ml-titanic/images/died.png"
        print("Survival predicted: Died.")
        survival_str = "died"
    img = Image.open(requests.get(survival_url, stream=True).raw)            
    img.save("./latest_survival.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_survival.png", "Resources/images-titanic", overwrite=True)
    
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()
    # print(df["survived"])
    label = df.iloc[-1]["survived"]
    if label == 1.0:
        label_url = "https://raw.githubusercontent.com/Hope-Liang/ID2223Lab1/main/serverless-ml-titanic/images/survived.png"
        print("Survival actual: Survived.")
        label_str = "survived"
    else:
        label_url = "https://raw.githubusercontent.com/Hope-Liang/ID2223Lab1/main/serverless-ml-titanic/images/died.png"
        print("Survival actual: Died.")
        label_str = "died"
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_survival.png")
    dataset_api.upload("./actual_survival.png", "Resources/images-titanic", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Survival Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survival_str],
        'label': [label_str],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images-titanic", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our titanic_predictions feature group has examples of all 2 titanic survivals
    print("Number of different survival predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Died', 'True Survived'],
                             ['Pred Died', 'Pred Survived'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images-titanic", overwrite=True)
    else:
        print("You need 2 different survival predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different titanic survival predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

