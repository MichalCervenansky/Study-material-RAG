import streamlit as st

st.title("Settings")
st.markdown("""
Configure the settings for the RAG service below.
""")

# Example setting
api_url = st.text_input("LangChain Service API URL:", value="http://localhost:8000")
if st.button("Save Settings"):
    st.success("Settings saved!")
