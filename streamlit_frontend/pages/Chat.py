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
            with requests.post(
                BACKEND_URL,
                json={"question": prompt},
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