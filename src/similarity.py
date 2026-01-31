from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_vector, job_vector) -> float:
    similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]

    return round(similarity_score * 100, 2)


if __name__ == "__main__":
    from feature_extraction import extract_tfidf_features

    resume = "python developer machine learning"
    job = "python developer nlp machine learning data analysis"

    r_vec, j_vec, _ = extract_tfidf_features(resume, job)

    score = calculate_similarity(r_vec, j_vec)
    print("Match score: ", score, "%")
