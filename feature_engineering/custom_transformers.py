# custom_transformers.py

import pandas as pd

# Function to map Yes/No to 1/0
def binary_label_encoder(df):
    return df.apply(lambda x: x.map({'Yes': 1, 'No': 0}))

# Function to map gender (Male/Female)
def gender_encoder(df):
    return df.apply(lambda x: x.map({'Male': 0, 'Female': 1}))
