# main.py
"""
Project: Car Fuel Efficiency Prediction
Author: Your Name
Description: 
    This script trains a Linear Regression model to predict a car's 
    fuel efficiency (Miles per Gallon - MPG) based on its Horsepower and Weight.
"""

# ==========================
# Import Libraries
# ==========================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# Suppress Warnings
# ==========================
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

# ==========================
# Load Dataset
# ==========================
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"
df = pd.read_csv(URL)

# ==========================
# Explore Data
# ==========================
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

print("\nFirst 5 Rows:")
print(df.head())

print("\nRandom 5 Rows:")
print(df.sample(5))

# ==========================
# Data Visualization
# ==========================
plt.scatter(df["Horsepower"], df["MPG"], alpha=0.5)
plt.xlabel("Horsepower")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Horsepower vs MPG")
plt.grid(True)
plt.savefig("scatter_hp_mpg.png")
plt.close()

plt.scatter(df["Weight"], df["MPG"], alpha=0.5, color="orange")
plt.xlabel("Weight")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Weight vs MPG")
plt.grid(True)
plt.savefig("scatter_weight_mpg.png")
plt.close()

# ==========================
# Define Features & Target
# ==========================
target = df["MPG"]
features = df[["Horsepower", "Weight"]]

# ==========================
# Train Linear Regression Model
# ==========================
lr = LinearRegression()
lr.fit(features, target)

# ==========================
# Model Evaluation
# ==========================
predictions = lr.predict(features)
mse = mean_squared_error(target, predictions)
r2 = r2_score(target, predictions)

print("\nModel Coefficients:", lr.coef_)
print("Model Intercept:", lr.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# ==========================
# Make a Prediction
# ==========================
sample_input = [[100, 2000]]
predicted_mpg = lr.predict(sample_input)
print(f"\nPredicted MPG for Horsepower=100 and Weight=2000: {predicted_mpg[0]:.2f}")
