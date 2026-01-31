def find_missing_keywords(resume_text: str, job_text: str) -> list:
    resume_words = set(resume_text.split())
    job_words = set(job_text.split())

    missing_keywords = job_words - resume_words
    return sorted(list(missing_keywords))


if __name__ == "__main__":
    resume = "python developer machine learning"
    job = "python developer nlp machine learning data analysis"

    missing = find_missing_keywords(resume, job)

    print("Missing keywords: ", missing)
