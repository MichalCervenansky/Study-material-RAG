import json
import requests
import streamlit as st
from pydantic import BaseModel
from typing import List, AsyncGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = f"{os.getenv('BACKEND_URL')}/query"

# Page config
st.set_page_config(page_title="RAG Chat", page_icon="💬")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with requests.post(
                BACKEND_URL,
                json={
                    "question": prompt,
                    "messages": st.session_state.messages[:-1]  # Send all messages except current prompt
                },
                stream=True,
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                content = data.get('answer', '')
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                continue
                
                message_placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            full_response = "Sorry, there was an error connecting to the backend service."
            message_placeholder.markdown(full_response)
    
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

class QueryRequest(BaseModel):
    question: str
    messages: List[dict] = []  # Add message history

async def rag_pipeline(document_store, query: str, messages: List[dict] = None) -> AsyncGenerator[str, None]:
    results = document_store.query_documents(query)
    
    if not results or not results['documents']:
        yield "No relevant documents found."
        return

    relevant_chunks = results['documents'][0]
    combined_context = " ".join(relevant_chunks)
    
    # Format chat history
    chat_history = ""
    if messages:
        chat_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])

    prompt = [
        {
            "role": "system",
            "content": f"""**RAG Assistant Guidelines**
            ...
            ### CHAT HISTORY ###
            {chat_history}

            ### CONTEXT ###
            {combined_context}

            ### QUESTION ###
            {query}"""
        },
        {
            "role": "user",
            "content": "Please provide a comprehensive answer with document citations."
        }
    ]