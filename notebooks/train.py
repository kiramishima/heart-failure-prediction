#!/usr/bin/env python
# coding: utf-8

import os
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report, root_mean_squared_error
import pickle

import warnings
warnings.filterwarnings('ignore')

# # Load Data

# Load Data
df = pl.read_csv("../DATASET/heart.csv")

# Preprocessing

# Encoder
le = LabelEncoder()
dv = DictVectorizer()

df = df.with_columns(
    Sex = le.fit_transform(df['Sex']),
    ChestPainType = le.fit_transform(df['ChestPainType']),
    RestingECG = le.fit_transform(df['RestingECG']),
    ExerciseAngina = le.fit_transform(df['ExerciseAngina']),
    ST_Slope = le.fit_transform(df['ST_Slope'])
)

# Split data
df_train_full, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

# Scaler
scaler = MinMaxScaler()

# Train
x_train = df_train.select(pl.exclude("HeartDisease"))
x_train = scaler.fit_transform(x_train) # Applying MinMaxScaler
x_train = pl.DataFrame(x_train, schema=df.select(pl.exclude("HeartDisease")).columns)
x_train = x_train.to_dicts()
x_train = dv.fit_transform(x_train)
x_train = csr_matrix(x_train)
y_train = df_train['HeartDisease']


# Validation
x_val = df_val.select(pl.exclude("HeartDisease"))
x_val = scaler.transform(x_val) # Applying MinMaxScaler
x_val = pl.DataFrame(x_val, schema=df.select(pl.exclude("HeartDisease")).columns)
x_val = x_val.to_dicts()
x_val = dv.transform(x_val)
x_val = csr_matrix(x_val)
y_val = df_val['HeartDisease']


# Test
x_test = df_test.select(pl.exclude("HeartDisease"))
x_test = scaler.transform(x_test) # Applying MinMaxScaler
x_test = pl.DataFrame(x_test, schema=df.select(pl.exclude("HeartDisease")).columns)
x_test = x_test.to_dicts()
x_test = dv.transform(x_test)
x_test = csr_matrix(x_test)
y_test = df_test['HeartDisease']


# Train models

# Logistic Regression
lr_model = LogisticRegression(random_state=42, solver='newton-cholesky')
lr_model.fit(x_train, y_train)

# DecisionTree

dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train, y_train)

# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(x_train, y_train)

# RandomForest
rfr = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=42, n_jobs=-1)
rfr.fit(x_train, y_train)

# Model Scores

models = [lr_model, dt, rfc, rfr]
models_names = ["LogisticRegression", "DecisionTree", "RandomForestClassifier", "RandomForestRegressor"]

# Scores
train_score = [model.score(x_train, y_train) for model in models]
test_score = [model.score(x_test, y_test) for model in models]
val_score = [model.score(x_val, y_val) for model in models]

# Measure model state:6
rate = []
for train, test, val in zip(train_score, test_score, val_score):
    if train <= 0.65 and test <= 0.65 and val <= 0.65:
        rate.append('bad')
    elif (train > 0.65 and train < 0.80) and (test > 0.65 and test < 0.80) and (val > 0.65 and val < 0.80):
        rate.append('middle')
    elif (train >= 0.80 and test >= 0.80 and val >= 0.80) and (train <= 0.999 and test <= 0.999 and val <= 0.999):
        rate.append('good') 
    else:
        rate.append('overfite')  # Handle cases that don't fit the above

# Create DataFrame
model_score = pl.DataFrame({
    'Model': models_names,
    'Train score': [f'{round(score * 100, 2)}%' for score in train_score],
    'Test score': [f'{round(score * 100, 2)}%' for score in test_score],
    'Val score': [f'{round(score * 100, 2)}%' for score in val_score],
    'Evaluate model': rate
})


# Saving models

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

dest_path = '../models'

for row in tqdm(model_score.filter(pl.col("Evaluate model") == "good").iter_rows()):
        model_name = row[0]
        idx = models_names.index(model_name)
        model = models[idx]
        dump_pickle(model, os.path.join(dest_path, f"{model_name}.pkl"))


# Save enconder and scaler
dump_pickle(le, os.path.join(dest_path, "encoder.bin"))
dump_pickle(scaler, os.path.join(dest_path, "scaler.bin"))
dump_pickle(dv, os.path.join(dest_path, "dv.bin"))

