import streamlit as st
import requests

# Set up the page configuration
st.set_page_config(page_title="RAG Service", page_icon="📘", layout="wide")

# Header
st.title("📘 Retrieval-Augmented Generation (RAG) Service")
st.markdown("Ask questions to get insights powered by LangChain and Ollama.")

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
