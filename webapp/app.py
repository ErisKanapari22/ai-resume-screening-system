import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extraction import extract_tfidf_features
from src.preprocessing import preprocess_text
from src.recommendations import find_missing_keywords
from src.similarity import calculate_similarity
from src.similarity import calculate_keyword_score

def run_pipeline(resume_text, job_text):
    # Preprocess
    resume_clean = preprocess_text(resume_text)
    job_clean = preprocess_text(job_text)

    # TF-IDF similarity
    resume_vec, job_vec, _ = extract_tfidf_features(resume_clean, job_clean)
    cosine_score = calculate_similarity(resume_vec, job_vec) * 100

    # Keyword score

    keyword_score = calculate_keyword_score(resume_clean, job_clean)

    # Final hybrid score
    final_score = 0.9 * keyword_score + 0.1 * cosine_score

    # Missing keywords
    missing_keywords = find_missing_keywords(resume_clean, job_clean)
    print("Keyword Score:", keyword_score)
    print("Cosine Score:", cosine_score)

    return {
        "score": round(final_score, 2),
        "missing_keywords": missing_keywords

    }

# UI configurations

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🚀 AI Resume Screener")
st.markdown("Upload your resume and compare it with a job description.")

# Input Section

col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Resume")
    uploaded_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
    resume_text = ""

    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8")

    resume_text = st.text_area("Or paste resume text", value=resume_text, height=300)

with col2:
    st.subheader("💼 Job Description")
    job_text = st.text_area("Paste job description", height=300)


# Analyze Button

if st.button("Analyze"):

    # Validation
    if not resume_text.strip() or not job_text.strip():
        st.error("Please provide both Resume and Job Description.")
    elif len(resume_text.split()) < 30 or len(job_text.split()) < 30:
        st.warning("Text too short. Please provide more detailed content.")
    else:
        result = run_pipeline(resume_text, job_text)

        score = result["score"]
        missing_keywords = result["missing_keywords"]

        st.divider()
        st.subheader("📊 Results")

        st.metric(label="Match Percentage", value=f"{score}%")

        st.progress(int(score))

        # Recommendation
        if score >= 70:
            st.success("✅ Good match")
        elif score < 50:
            st.error("❌ Needs improvement")
        else:
            st.info("No Match!")

        # Missing Keywords
        st.subheader("🔑 Missing Keywords")

        if missing_keywords:
            st.write(", ".join(missing_keywords))
        else:
            st.success("No missing keywords 🎉")


