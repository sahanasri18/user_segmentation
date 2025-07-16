# model_training.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

# Load dataset
df = pd.read_csv('data/users.csv')

# Optional: Encode gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Select features
features = ['Age', 'Gender', 'Income', 'SpendingScore', 'Purchases']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open('models/kmeans_model.pkl', 'wb') as f:
    pickle.dump((scaler, kmeans), f)

print("✅ Model trained and saved to models/kmeans_model.pkl")
