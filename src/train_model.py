import pandas as pd
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("/content/IMDB Dataset.csv")

df["cleaned_review"] = df["review"].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred, average="weighted")
print("F1-score:", round(f1, 2))
print(classification_report(y_test, y_pred))
