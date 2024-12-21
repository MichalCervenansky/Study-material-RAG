import streamlit as st
import requests

# Set up the page configuration
st.set_page_config(page_title="RAG Service", page_icon="📘", layout="wide")

# Header
st.title("📘 Retrieval-Augmented Generation (RAG) Service")
st.markdown("Upload your study materials as PDFs and ask questions to get insights powered by LangChain and Ollama.")

# File uploader for PDFs
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# Display uploaded files
if uploaded_files:
    st.subheader("Uploaded Files")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")

    # Submit button
    if st.button("Process Documents"):
        st.info("Processing documents... This may take a moment.")
        # Make a request to the LangChain service
        for uploaded_file in uploaded_files:
            response = requests.post(
                "http://localhost:8000/upload",
                files={"file": uploaded_file.getvalue()}
            )
            if response.status_code == 200:
                st.success(f"Processed: {uploaded_file.name}")
            else:
                st.error(f"Failed to process: {uploaded_file.name}")

# Question input
st.subheader("Ask a Question")
question = st.text_input("Type your question here:")

if question:
    # Make a request to the LangChain service
    response = requests.post(
        "http://localhost:8000/query",
        json={"question": question}
    )
    if response.status_code == 200:
        answer = response.json().get("answer", "No answer found.")
        st.write(f"**Answer:** {answer}")
    else:
        st.error("Failed to fetch an answer. Please try again.")
