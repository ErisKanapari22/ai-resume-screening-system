from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(resume_text: str, job_text: str):
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

    return tfidf_matrix[0], tfidf_matrix[1], vectorizer