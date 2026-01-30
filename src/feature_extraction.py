from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(resume_text: str, job_text: str):

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

    resume_vector = tfidf_matrix[0]
    job_vector = tfidf_matrix[1]

    return resume_vector, job_vector, vectorizer

if __name__ == "__main__":
    resume = "python developer machine learning"
    job = "python developer nlp machinelearning data analysis"

    r_vec, j_vec, vec = extract_tfidf_features(resume, job)

    print(f"Resume Vector shape: {r_vec.shape}")
    print(f"Job Vector shape: {j_vec.shape}")


