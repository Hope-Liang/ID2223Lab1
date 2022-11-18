import pandas as pd 
from preprocessor_pipeline import preprocessing_titanic

titanic_df = pd.read_csv("../../Titanic/train.csv")
titanic_df = preprocessing_titanic(titanic_df)
print(titanic_df.columns)
titanic_df_1 = titanic_df[titanic_df.Survived == 1]
titanic_df_0 = titanic_df[titanic_df.Survived == 0]
ttl_1 = titanic_df_1.shape[0]
ttl_0 = titanic_df_0.shape[0]

male_1 = titanic_df_1[titanic_df_1.Sex_male == 1].shape[0]
print("p_male_survived = ", male_1/ttl_1)
male_0 = titanic_df_0[titanic_df_0.Sex_male == 1].shape[0]
print("p_male_died = ", male_0/ttl_0)

embark_c_1 = titanic_df_1[titanic_df_1.Embarked_C == 1].shape[0]
print("p_embark_c_survived = ", embark_c_1/ttl_1)
embark_q_1 = titanic_df_1[titanic_df_1.Embarked_Q == 1].shape[0]
print("p_embark_q_survived = ", embark_q_1/ttl_1)
embark_s_1 = titanic_df_1[titanic_df_1.Embarked_S == 1].shape[0]
print("p_embark_s_survived = ", embark_s_1/ttl_1)

embark_c_0 = titanic_df_0[titanic_df_0.Embarked_C == 1].shape[0]
print("p_embark_c_died = ", embark_c_0/ttl_0)
embark_q_0 = titanic_df_0[titanic_df_0.Embarked_Q == 1].shape[0]
print("p_embark_q_died = ", embark_q_0/ttl_0)
embark_s_0 = titanic_df_0[titanic_df_0.Embarked_S == 1].shape[0]
print("p_embark_s_died = ", embark_s_0/ttl_0)

age_1_1 = titanic_df_1[titanic_df_1.Age_Group == 1].shape[0]
print("p_age_1_survived = ", age_1_1/ttl_1)
age_2_1 = titanic_df_1[titanic_df_1.Age_Group == 2].shape[0]
print("p_age_2_survived = ", age_2_1/ttl_1)
age_3_1 = titanic_df_1[titanic_df_1.Age_Group == 3].shape[0]
print("p_age_3_survived = ", age_3_1/ttl_1)

age_1_0 = titanic_df_0[titanic_df_0.Age_Group == 1].shape[0]
print("p_age_1_died = ", age_1_0/ttl_0)
age_2_0 = titanic_df_0[titanic_df_0.Age_Group == 2].shape[0]
print("p_age_2_died = ", age_2_0/ttl_0)
age_3_0 = titanic_df_0[titanic_df_0.Age_Group == 3].shape[0]
print("p_age_3_died = ", age_3_0/ttl_0)

print("max_sibsp_survived =", titanic_df_1["SibSp"].max())
print("max_sibsp_died =", titanic_df_0["SibSp"].max())

print("max_parch_survived =", titanic_df_1["Parch"].max())
print("max_parch_died =", titanic_df_0["Parch"].max())

print("max_fare_survived =", titanic_df_1["Fare"].max())
print("min_fare_survived =", titanic_df_1["Fare"].min())
print("max_parch_died =", titanic_df_0["Fare"].max())
print("max_parch_died =", titanic_df_0["Fare"].min())

pclass_1_1 = titanic_df_1[titanic_df_1.Pclass == 1].shape[0]
print("pclass_1_survived = ", pclass_1_1/ttl_1)
pclass_2_1 = titanic_df_1[titanic_df_1.Pclass == 2].shape[0]
print("pclass_2_survived = ", pclass_2_1/ttl_1)
pclass_3_1 = titanic_df_1[titanic_df_1.Pclass == 3].shape[0]
print("pclass_3_survived = ", pclass_3_1/ttl_1)

pclass_1_0 = titanic_df_0[titanic_df_0.Pclass == 1].shape[0]
print("pclass_1_died = ", pclass_1_0/ttl_0)
pclass_2_0 = titanic_df_0[titanic_df_0.Pclass == 2].shape[0]
print("pclass_2_died = ", pclass_2_0/ttl_0)
pclass_3_0 = titanic_df_0[titanic_df_0.Pclass == 3].shape[0]
print("pclass_3_died = ", pclass_3_0/ttl_0)
