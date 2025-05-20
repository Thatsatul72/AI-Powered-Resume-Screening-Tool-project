import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import nltk
import re
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download NLTK data only if not already downloaded
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Set up Streamlit page
st.set_page_config(page_title="BERT Resume Matcher", layout="wide")
st.title("🤖 AI Resume Matcher using BERT")
st.markdown("Upload resumes and a job description — see similarity scores using **semantic NLP** (Sentence-BERT).")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract keywords from text
def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [re.sub(r'\W+', '', word) for word in tokens if word.isalpha()]
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return set(keywords)

# Upload UI
uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
job_desc = st.text_area("📝 Paste Job Description Here", height=200)

if st.button("🚀 Match Resumes"):
    if uploaded_files and job_desc.strip():
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            try:
                text = extract_text_from_pdf(file)
                resume_texts.append(text)
                resume_names.append(file.name)
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")

        # Load Sentence-BERT model
        with st.spinner("🔍 Computing semantic similarity..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Encode job description and resumes
            all_docs = [job_desc] + resume_texts
            embeddings = model.encode(all_docs, convert_to_tensor=True)

            # Compute cosine similarity
            job_embedding = embeddings[0]
            resume_embeddings = embeddings[1:]
            scores = util.cos_sim(job_embedding, resume_embeddings).flatten().tolist()

            # Extract job description keywords
            job_keywords = extract_keywords(job_desc)
            results = []

            for i in range(len(resume_texts)):
                resume_keywords = extract_keywords(resume_texts[i])
                matched = job_keywords & resume_keywords
                missing = job_keywords - resume_keywords

                results.append({
                    "Resume": resume_names[i],
                    "Match Score (0–100)": round(scores[i] * 100, 2),
                    "Matched Keywords": ", ".join(sorted(matched)),
                    "Missing Keywords": ", ".join(sorted(missing))
                })

            results_df = pd.DataFrame(results).sort_values(by="Match Score (0–100)", ascending=False).reset_index(drop=True)

        st.success("✅ Matching complete!")
        st.dataframe(results_df)

        # Download as CSV
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "resume_match_results.csv", "text/csv")
    else:
        st.warning("⚠️ Please upload resumes and enter a job description before matching.")
