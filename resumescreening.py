import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("📄 AI-Powered Resume Screening Tool")
st.markdown("""
Upload multiple **resumes in PDF format**, enter the **job description**, and get a ranked list of resumes based on NLP similarity using TF-IDF.
""")

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# UI for file upload and job description
uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
job_desc = st.text_area("📝 Paste Job Description", height=200)

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
                st.error(f"Error processing {file.name}: {str(e)}")

        # TF-IDF vectorization
        documents = [job_desc] + resume_texts
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Cosine similarity
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Results in DataFrame
        results = pd.DataFrame({
            "Resume": resume_names,
            "Match Score": scores
        }).sort_values(by="Match Score", ascending=False).reset_index(drop=True)

        st.success("🎉 Matching complete!")
        st.dataframe(results)

        # Download CSV
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "resume_match_results.csv", "text/csv")
    else:
        st.warning("Please upload resumes and enter a job description before matching.")
