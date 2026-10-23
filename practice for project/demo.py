import re
import joblib


model = joblib.load("flag_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
  
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_review(review_text):
    cleaned = clean_text(review_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return prediction

print("=== Negative Review Flagging Demo ===")
print("Type a review, or 'quit' to exit.\n")

while True:
    review = input("Enter a review: ")
    if review.lower() == "quit":
        break
    result = classify_review(review)
    if result == 1:
        print(">> FLAGGED - route to a support agent.\n")
    else:
        print(">> Normal - no action needed.\n")

print("Goodbye.")