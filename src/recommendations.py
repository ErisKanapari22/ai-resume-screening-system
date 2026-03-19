from src.keyword_extraction import extract_keywords


def find_missing_keywords(resume_text: str, job_text: str) -> list:
    resume_keywords = set(extract_keywords(resume_text))
    job_keywords = set(extract_keywords(job_text))

    missing = job_keywords - resume_keywords

    return sorted(list(missing))


if __name__ == "__main__":
    resume = "python developer machine learning"
    job = "python developer nlp machine learning data analysis"

    missing = find_missing_keywords(resume, job)

    print("Missing keywords: ", missing)
