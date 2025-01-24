import json
import requests
import streamlit as st

# Backend configuration
BACKEND_URL = "http://localhost:8000/query"

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
            # Make streaming request
            with requests.post(
                BACKEND_URL,
                json={"question": prompt},
                stream=True,
                headers={"Accept": "text/event-stream"}
            ) as response:
                response.raise_for_status()
                
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = line.decode('utf-8')
                            # Handle the case where the response is a generator
                            if chunk.startswith('data: '):
                                chunk = chunk[6:]  # Remove 'data: ' prefix
                            chunk_data = json.loads(chunk)
                            if isinstance(chunk_data, str):
                                content = chunk_data
                            elif "answer" in chunk_data:
                                content = chunk_data["answer"]
                            else:
                                continue
                                
                            if isinstance(content, list):
                                content = "".join(content)
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue

            # Final update without cursor
            if full_response:
                message_placeholder.markdown(full_response)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to backend: {str(e)}"
            st.error(error_msg)
            full_response = "Sorry, there was an error connecting to the backend service."
            message_placeholder.markdown(full_response)
    
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()