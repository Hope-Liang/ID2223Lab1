import os
import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(survived, p_male, p_embark_c, p_embark_q, p_age_1, p_age_2, max_sibsp, max_parch, fare_min, fare_max, p_class_1, p_class_2):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    sex_rand = random.uniform(0,1)
    if sex_rand < p_male:
        sex_female = 0.0
        sex_male = 1.0
    else:
        sex_female = 1.0
        sex_male = 0.0

    embark_rand = random.uniform(0,1)
    if embark_rand < p_embark_c:
        embarked_c = 1.0
        embarked_q = 0.0
        embarked_s = 0.0
    elif embarked_rand < p_embark_c+p_embark_q:
        embarked_c = 0.0
        embarked_q = 1.0
        embarked_s = 0.0
    else:
        embarked_c = 0.0
        embarked_q = 0.0
        embarked_s = 1.0

    age_rand = random.uniform(0,1)
    if age_rand < p_age_1:
        age_group = 1.0
    elif age_rand < p_age_1 + p_age_2:
        age_group = 2.0
    else:
        age_group = 3.0

    pclass_rand = random.uniform(0,1)
    if pclass_rand < p_class_1:
        pclass = 1.0
    elif pclass_rand < p_class_1 + p_class_2:
        pclass = 2.0
    else:
        pclass = 3.0

    df = pd.DataFrame({"Sex_female": [sex_female],
                        "Sex_male": [sex_male],
                        "Embarked_C": [embarked_c],
                        "Embarked_Q": [embarked_q],
                        "Embarked_S": [embarked_s],
                        "Age_Group": [age_group],
                        "Sibsp": [float(random.randint(0,max_sibsp))],
                        "Parch": [float(random.randint(0,max_parch))],
                        "Fare": [random.uniform(fare_max, fare_min)],
                        "Pclass": [pclass]
                    })
    df['Survived'] = survived
    return df


def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random titanic passenger
    """
    import pandas as pd
    import random

    survived_df = generate_passenger(1.0, p_male=0.3187, p_embark_c=0.2719, p_embark_q=0.0877, p_age_1=0.1784, p_age_2=0.7836, 
                                    max_sibsp=4.0, max_parch=5.0, fare_min=0.0, fare_max=512.3292, p_class_1=0.3977, p_class_2=0.2544)
    died_df = generate_passenger(0.0, p_male=0.8525, p_embark_c=0.1366, p_embark_q=0.0856, p_age_1=0.0947, p_age_2=0.8525, 
                                    max_sibsp=8.0, max_parch=6.0, fare_min=0.0, fare_max=263.0, p_class_1=0.1457, p_class_2=0.1767)

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        titanic_df = survived_df
        print("Survived added")
    else:
        titanic_df = died_df
        print("Died added")

    return titanic_df



def g():
    import hopsworks
    import pandas as pd
    from preprocessor_pipeline import preprocessing_titanic

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
        titanic_df = preprocessing_titanic(titanic_df)
    else:
        titanic_df = get_random_titanic_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_Group', 'SibSp', 'Parch', 'Fare', 'Pclass'], 
        description="Titanic survival dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})



if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
