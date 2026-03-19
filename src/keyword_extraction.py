import re

TECH_KEYWORDS = {
    "python", "java", "javascript", "sql",
    "mysql", "postgresql",
    "django", "flask", "fastapi",
    "docker", "kubernetes",
    "aws", "azure", "gcp",
    "machine learning", "nlp",
    "api", "rest",
    "linux", "git"
}


SYNONYMS = {
    "ml": "machine learning",
    "machine-learning": "machine learning"
}


def normalize_text(text: str) -> str:
    text = text.lower()

    # Replace synonyms
    for key, value in SYNONYMS.items():
        text = re.sub(rf"\b{key}\b", value, text)

    return text


def extract_keywords(text: str) -> list:
    text = normalize_text(text)

    found_keywords = set()

    for keyword in TECH_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            found_keywords.add(keyword)

    return sorted(found_keywords)