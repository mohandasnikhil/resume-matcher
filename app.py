import streamlit as st
import os
import tempfile
import pandas as pd
from pdfminer.high_level import extract_text as extract_pdf
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import re
from difflib import get_close_matches

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            text = extract_pdf(tmp.name)
        return text
    elif ext == '.docx':
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_name_from_text(text, names_list):
    lines = text.strip().split("\n")
    for line in lines[:5]:
        match = get_close_matches(line.strip().lower(), [n.lower() for n in names_list], n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

def safe_to_numpy(embedding):
    if isinstance(embedding, torch.Tensor):
        return embedding.detach().cpu().numpy()
    return embedding

def get_similarity(text, job_text):
    emb_resume = model.encode(text, convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)
    emb_resume_np = safe_to_numpy(emb_resume)
    emb_job_np = safe_to_numpy(emb_job)
    sim = cosine_similarity([emb_resume_np], [emb_job_np])[0][0]
    return round(sim * 10, 2)

def extract_skills_from_jd(text):
    keywords = re.findall(r"\b(?:Python|NLP|machine learning|communication|data|SQL|deep learning|analytics|modeling|cloud|statistics|leadership|presentation|research|Excel|Tableau)\b", text, flags=re.I)
    return list(set(k.lower() for k in keywords))

def skill_match_summary(candidate_text, jd_skills):
    found = []
    missing = []
    text = candidate_text.lower()
    for skill in jd_skills:
        if skill in text:
            found.append(skill)
        else:
            missing.append(skill)
    return found, missing

def parse_answers(answer_str):
    parts = [a.strip() for a in str(answer_str).split('|')]
    while len(parts) < 6:
        parts.append("")
    return parts

def score_salary(salaries):
    salaries = np.array(salaries, dtype=float)
    max_val, min_val = salaries.max(), salaries.min()
    return [round((max_val - s) / (max_val - min_val + 1e-5) * 10, 2) for s in salaries]

def score_notice_period(periods):
    values = []
    for p in periods:
        match = re.search(r"\d+", str(p))
        values.append(int(match.group()) if match else 60)
    values = np.array(values)
    max_val, min_val = values.max(), values.min()
    return [round((max_val - v) / (max_val - min_val + 1e-5) * 10, 2) for v in values]

# --- Streamlit App Starts Here ---

st.title("📊 Smart Resume Screener")
st.write("Paste the job description and upload resumes below:")

# 1. Job description input
job_text = st.text_area("📝 Job Description", height=200)

# 2. Upload LinkedIn responses (optional)
responses_file = st.file_uploader("📎 Upload LinkedIn Answers CSV (Optional)", type=["csv"])
responses_df = pd.read_csv(responses_file) if responses_file else None
response_names = responses_df["name"].tolist() if responses_df is not None else []

# 3. Upload resumes
uploaded_files = st.file_uploader("📄 Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and job_text.strip():
    jd_skills = extract_skills_from_jd(job_text)
    results = []

    for file in uploaded_files:
        resume_text = extract_text(file)

        if responses_df is not None:
            matched_name = extract_name_from_text(resume_text, response_names)
            if not matched_name:
                st.warning(f"❌ Could not match resume to any name in uploaded CSV: {file.name}")
                continue

            match = responses_df[responses_df['name'].str.lower() == matched_name.lower()]
            if match.empty:
                st.warning(f"❌ No matching record in uploaded CSV for: {matched_name}")
                continue

            answers = parse_answers(match.iloc[0]['answers'])
            if answers[0].lower() != "yes" or answers[1].lower() != "yes":
                continue  # Skip ineligible

            experience = answers[2]
            education = answers[3]
            salary = float(re.sub(r"[^\d.]", "", answers[4]) or 0)
            notice = answers[5]
            full_text = resume_text + "\n" + experience + "\n" + education
            primary_score = get_similarity(full_text, job_text)
            matched, missing = skill_match_summary(full_text, jd_skills)

            results.append({
                "name": matched_name.title(),
                "score": primary_score,
                "salary": salary,
                "notice": notice,
                "skills_matched": ", ".join(matched),
                "skills_missing": ", ".join(missing)
            })

        else:
            # Resume-only mode
            name = file.name.replace("_", " ").replace(".pdf", "").replace(".docx", "")
            primary_score = get_similarity(resume_text, job_text)
            matched, missing = skill_match_summary(resume_text, jd_skills)

            results.append({
                "name": name.title(),
                "score": primary_score,
                "salary": "N/A",
                "notice": "N/A",
                "skills_matched": ", ".join(matched),
                "skills_missing": ", ".join(missing)
            })

    if results:
        df = pd.DataFrame(results)
        if responses_df is not None:
            df["salary_score"] = score_salary(df["salary"])
            df["notice_score"] = score_notice_period(df["notice"])
            df["final_rank"] = (df["score"] * 0.7 + df["salary_score"] * 0.2 + df["notice_score"] * 0.1).round(2)
        else:
            df["final_rank"] = df["score"]

        st.subheader("✅ Ranked Candidates")
        sorted_df = df.sort_values("final_rank", ascending=False)
        st.dataframe(sorted_df, use_container_width=True)

        csv = sorted_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="ranked_candidates.csv",
            mime="text/csv"
        )
    else:
        st.warning("No eligible candidates matched.")
elif not job_text.strip():
    st.info("Please enter a job description to continue.")
