import streamlit as st
# from Models.BERT_model_pipeline import BERT_model
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
        # Cybermodel = BERT_model()
        # Cybermodel.load_model()  
        
        user_input = st.text_area("Enter your text here:", height=100)
     
        with st.expander("ℹ️ About this system"):
            st.write("""
            This system uses a BERT-based deep learning model to detect potential cyberbullying content in text. 
            The model has been trained on a dataset of labeled examples to recognize patterns associated with cyberbullying.
            
            - **Not Bullying**: Text that appears safe and non-harmful
            - **Bullying**: Text that contains potential cyberbullying content
            
            Please note that this is an automated system and should be used as a tool to assist in content moderation, 
            not as the sole decision maker.
            """)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure the model is properly loaded and all dependencies are installed.")

if __name__ == "__main__":
    main()