import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("archive/Cleaned_data.csv")

X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']

ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['location'])
], remainder='passthrough')

pipe = Pipeline([
    ('transform', ct),
    ('model', Ridge())
])

pipe.fit(X, y)

pickle.dump(pipe, open("archive/RidgeModel.pkl", "wb"))

print("New model created successfully")