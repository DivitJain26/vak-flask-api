import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib

df = pd.read_csv("train.csv")

df["label"] = (
    (df["toxic"] >= 0.5)
    | (df["severe_toxic"] >= 0.5)
    | (df["obscene"] >= 0.5)
    | (df["threat"] >= 0.5)
    | (df["insult"] >= 0.5)
    | (df["identity_hate"] >= 0.5)
).astype(int)


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df.rename(columns={"comment_text": "text"}, inplace=True)
df["clean_text"] = df["text"].apply(preprocess)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

model = LinearSVC()
model.fit(X, y)

joblib.dump(model, "vak_svm_model.pkl")
joblib.dump(tfidf, "vak_tfidf.pkl")

print("Model and TF-IDF saved!")
