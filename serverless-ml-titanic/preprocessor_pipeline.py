import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline
import numpy as np

titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

titanic_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Turn 'Age' column into groups
age_conditions = [
    (titanic_df['Age'] < 18),
    (titanic_df['Age'] >= 18) & (titanic_df['Age'] < 55),
    (titanic_df['Age'] >= 55),
]

age_groups = [1, 2, 3]

titanic_df['Age_Group'] = np.select(age_conditions, age_groups)
titanic_df.drop(columns=['Age'], inplace=True)
titanic_df['Age_Group'].replace(0, np.nan, inplace=True)

# Pre-Processor #

# Column Transformer Settings
mode_impute_onehot_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                            OneHotEncoder(sparse=False, handle_unknown='ignore')
                                            )

t = [('do_nothing', SimpleImputer(strategy='most_frequent'), ['PassengerId']),
     ('ohe-cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), ['Sex']),
     ('mode-impute', mode_impute_onehot_pipeline, ['Embarked']),
     ('mode', SimpleImputer(strategy='most_frequent'), ['Age_Group']),
     ('do_nothing2', SimpleImputer(strategy='most_frequent'), ['SibSp', 'Parch', 'Fare', 'Pclass', 'Survived'])
     ]
col_transformer = ColumnTransformer(transformers=t, remainder='drop')

pre_processor = col_transformer
#############################

titanic_df_processed = pre_processor.fit_transform(X=titanic_df)


def get_feature_out(estimator, feature_in):
    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                    for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == 'passthrough':
            output_features.extend(ct._feature_names_in[features])

    return output_features


# Get column names
columns = get_ct_feature_names(pre_processor)
titanic_df_processed = pd.DataFrame(titanic_df_processed, columns=columns)
