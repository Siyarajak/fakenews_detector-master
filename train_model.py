# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load datasets
fake = pd.read_csv(r"C:\Users\siyar\Downloads\archive (8)\Fake.csv")
true = pd.read_csv(r"C:\Users\siyar\Downloads\archive (8)\True.csv")

# Label the data
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Split
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
