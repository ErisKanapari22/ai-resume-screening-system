import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    cleaned_words = [word for word in words if word not in STOP_WORDS]

    return " ".join(cleaned_words)


if __name__ == "__main__":
    sample = "Python developer with experience in Machine Learning!"
    print(preprocess_text(sample))