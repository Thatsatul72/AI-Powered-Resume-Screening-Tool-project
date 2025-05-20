
# AI Resume Matcher using BERT

A simple and interactive web app that leverages **Sentence-BERT** embeddings to match candidate resumes against a job description using semantic similarity and keyword matching.

---

## Features

- Upload multiple PDF resumes at once  
- Paste or type a job description  
- Extract text content from PDFs using PyMuPDF (`fitz`)  
- Compute semantic similarity scores between job description and resumes using Sentence-BERT (`all-MiniLM-L6-v2`)  
- Extract and compare keywords between job description and resumes using NLTK  
- Display matched and missing keywords per resume  
- View and sort results in an interactive table  
- Download the matching results as a CSV file  

---

## Tech Stack

- Python  
- Streamlit for the web UI  
- PyMuPDF (`fitz`) for PDF text extraction  
- Sentence-Transformers (Sentence-BERT) for embeddings and similarity  
- NLTK for text preprocessing and keyword extraction  
- Pandas for tabular data handling  

---

## Installation

1. Clone the repository or copy the code.  
2. Create a Python virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
