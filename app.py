import streamlit as st
import os
import tempfile
import pandas as pd
import zipfile
from io import BytesIO
from pdfminer.high_level import extract_text as extract_pdf
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import re
from difflib import get_close_matches

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# Resume parsing
def extract_text(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes.read())
        tmp_path = tmp.name
    if ext == '.pdf':
        return extract_pdf(tmp_path)
    elif ext == '.docx':
        doc = Document(tmp_path)
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
    return round(cosine_similarity([safe_to_numpy(emb_resume)], [safe_to_numpy(emb_job)])[0][0] * 10, 2)

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

# UI starts
st.title("📂 Batch Resume Screener")
st.write("Upload a job description, a CSV of candidate answers (optional), and a ZIP file of resumes.")

job_text = st.text_area("📝 Job Description", height=200)

responses_file = st.file_uploader("📎 Upload LinkedIn Answers CSV (Optional)", type=["csv"])
responses_df = pd.read_csv(responses_file) if responses_file else None
response_names = responses_df["name"].tolist() if responses_df is not None else []

zip_file = st.file_uploader("📄 Upload Resumes (ZIP of PDFs/DOCXs)", type=["zip"])

if zip_file and job_text.strip():
    jd_skills = extract_skills_from_jd(job_text)
    results = []

    with zipfile.ZipFile(zip_file) as archive:
        resume_files = [f for f in archive.namelist() if f.endswith(('.pdf', '.docx'))]

        for filename in resume_files:
            file_bytes = BytesIO(archive.read(filename))
            resume_text = extract_text(file_bytes, filename)

            # Resume-only fallback
            if responses_df is None:
                name = resume_text.strip().split('\n')[0].strip()
                score = get_similarity(resume_text, job_text)
                matched, missing = skill_match_summary(resume_text, jd_skills)

                results.append({
                    "name": name,
                    "score": score,
                    "salary": "N/A",
                    "notice": "N/A",
                    "skills_matched": ", ".join(matched),
                    "skills_missing": ", ".join(missing)
                })

            else:
                matched_name = extract_name_from_text(resume_text, response_names)
                if not matched_name:
                    st.warning(f"❌ Could not match resume to any name in uploaded CSV: {filename}")
                    continue

                match = responses_df[responses_df['name'].str.lower() == matched_name.lower()]
                if match.empty:
                    st.warning(f"❌ No match in CSV for: {matched_name}")
                    continue

                answers = parse_answers(match.iloc[0]['answers'])
                if answers[0].lower() != "yes" or answers[1].lower() != "yes":
                    continue

                experience = answers[2]
                education = answers[3]
                salary = float(re.sub(r"[^\d.]", "", answers[4]) or 0)
                notice = answers[5]
                full_text = resume_text + "\n" + experience + "\n" + education
                score = get_similarity(full_text, job_text)
                matched, missing = skill_match_summary(full_text, jd_skills)

                results.append({
                    "name": matched_name.title(),
                    "score": score,
                    "salary": salary,
                    "notice": notice,
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
