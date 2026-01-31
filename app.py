from src.preprocessing import preprocess_text
from src.feature_extraction import extract_tfidf_features
from src.similarity import calculate_similarity
from src.recommendations import find_missing_keywords

def load_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    resume_path = "C:\\Users\\erisk\\Desktop\\PythonProjects\\ai-resume-screening-system\\data\\resumes\\sample_resume.txt"
    job_path = "C:\\Users\\erisk\\Desktop\\PythonProjects\\ai-resume-screening-system\\data\\job_descriptions\\sample_jd.txt"

    resume_text = load_text(resume_path)
    job_text = load_text(job_path)

    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_text)

    resume_vec, job_vec, _ = extract_tfidf_features(clean_resume, clean_job)

    match_percentage = calculate_similarity(resume_vec, job_vec)

    missing_keywords = find_missing_keywords(clean_resume, clean_job)

    print("\nAI Resume Screener Results")
    print("-" * 40)
    print(f"Match Percentage: {match_percentage}%")

    if missing_keywords:
        print("\nMissing Keywords")
        for keyword in missing_keywords:
            print(f"- {keyword}")

    else:
        print("\nResume Matches the Job description")

if __name__ == "__main__":
    main()

