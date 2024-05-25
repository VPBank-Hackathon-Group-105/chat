import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("Welcome! :grinning:") 
st.markdown("## :white_check_mark: About this product")

st.markdown("""This is a demo product of team 105 in the VPBank hackathon contest.""")
st.markdown("""

## :question: How to use:
    1. Go to the 'Upload CV' page.
    2. Upload CV files in one of the following formats: docx, doc, pdf, xlsx.
    3. Enter the chatbot and feel free to try it out. You can ask anything about the CV.
""")