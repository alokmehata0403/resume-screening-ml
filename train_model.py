import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset", "candidate_job_role_dataset.csv")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# Consistent text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', ' ', str(text))
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Load dataset
df = pd.read_csv(dataset_path)

# Combine features & clean
df["text"] = (
    df["skills"].astype(str) + " " +
    df["qualification"].astype(str) + " " +
    df["experience_level"].astype(str)
).apply(clean_text)

X = df["text"]
y = df["job_role"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
X_vector = tfidf.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y_encoded, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Test Accuracy: {acc * 100:.2f}%")

# Save models safely
with open(os.path.join(model_dir, "tfidf.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(model_dir, "clf.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

print("Model trained and saved successfully in:", model_dir)
