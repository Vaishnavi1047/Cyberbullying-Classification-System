import streamlit as st
from Models.BERT_model_pipeline import BERT_model
import torch
import asyncio
import os

if os.name == "nt": 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main():
    st.set_page_config(
        page_title="Cyberbullying Detection",
        page_icon="🛡️",
        layout="centered"
    )

    st.title("Cyberbullying Detection System ")
    st.write("Enter a message to check if it contains cyberbullying content.")
    
    try:
        Cybermodel = BERT_model()
        Cybermodel.load_model()  
        
        user_input = st.text_area("Enter your text here:", height=100)
     
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure the model is properly loaded and all dependencies are installed.")

if __name__ == "__main__":
    main()