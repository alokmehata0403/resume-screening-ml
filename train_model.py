import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset/candidate_job_role_dataset.csv")

# Combine features into one text column
df["text"] = (
    df["skills"].astype(str) + " " +
    df["qualification"].astype(str) + " " +
    df["experience_level"].astype(str)
)

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

# Save models
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(model, open("model/clf.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))

print("Model trained and saved successfully!")