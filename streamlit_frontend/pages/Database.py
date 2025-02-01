import streamlit as st
import os
from PyPDF2 import PdfReader
import zipfile
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import requests

BACKEND_URL = "http://localhost:8000"

def make_request(endpoint: str, method: str = "GET", json_data: dict = None):
    try:
        url = f"{BACKEND_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file) -> str:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_zip_file(zip_file) -> List[str]:
    texts = []
    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            if filename.endswith('.pdf'):
                with z.open(filename) as f:
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(f.read())
                        temp_file.flush()
                        texts.append(extract_text_from_pdf(temp_file.name))
                    os.unlink(temp_file.name)
    return texts

def chunk_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

st.title("Document Database Management")

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose PDF or ZIP files", type=['pdf', 'zip'], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    processed_files = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
                texts = [text]
            else:  # ZIP file
                texts = process_zip_file(uploaded_file)
            
            for text in texts:
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                # Simplified metadata
                processed_files.extend([{
                    "source": uploaded_file.name
                } for _ in range(len(chunks))])
            
            st.success(f"Processed {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if all_chunks:
        st.info(f"Total files: {len(uploaded_files)} | Total chunks: {len(all_chunks)}")
            
        if st.button("Ingest to ChromaDB"):
            response = make_request(
                "documents/add",
                method="POST",
                json_data={
                    "documents": all_chunks,
                    "metadatas": processed_files,
                    "ids": [f"doc_{i}" for i in range(len(all_chunks))]
                }
            )
            if response and response["status"] == "success":
                st.success(f"Successfully ingested {len(all_chunks)} chunks!")
            else:
                st.error("Failed to ingest documents")

# Document listing section
st.header("Stored Documents")
if st.button("List Documents"):
    results = make_request("documents")
    if results and results.get('documents'):
        df_data = {
            'Source': [m.get('source', 'Unknown') for m in results['metadatas']],
            'Content': results['documents']
        }
        
        st.dataframe(
            df_data,
            column_config={
                'Source': st.column_config.TextColumn('Source File'),
                'Content': st.column_config.TextColumn('Content', width='large')
            },
            hide_index=True
        )
        
        st.info(f"Total chunks: {len(results['documents'])}")
    else:
        st.info("No documents found in the database.")

# Add clear database option
if st.button("Clear Database"):
    response = make_request("documents/clear", method="POST")
    if response and response["status"] == "success":
        st.success("Database cleared successfully!")
    else:
        st.error("Failed to clear database")
