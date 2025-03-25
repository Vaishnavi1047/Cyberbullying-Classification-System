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
        
        if st.button("Analyze Text"):
            if user_input.strip() != "":
                with st.spinner('Analyzing text...'):
                    encoded = Cybermodel.tokenizer(
                        user_input,
                        add_special_tokens=True,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    ).to(Cybermodel.device)
                    
                    Cybermodel.model.eval()
                    with torch.no_grad():
                        outputs = Cybermodel.model(**encoded)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class = torch.argmax(predictions, dim=-1).item()
                        confidence = predictions[0][predicted_class].item()
                    
                    st.write("---")
                    st.write("### Analysis Results:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Classification:**")
                        if predicted_class == 0:
                            st.success("Not Bullying")
                        else:
                            st.error("Bullying Detected")
                    
                    with col2:
                        st.write("**Confidence:**")
                        st.write(f"{confidence:.2%}")
                    
                    st.write("---")
                    st.write("### Interpretation:")
                    if predicted_class == 1:
                        if confidence > 0.90:
                            st.warning("This text shows strong indicators of cyberbullying content.")
                        else:
                            st.warning("This text shows some indicators of cyberbullying content.")
                    else:
                        if confidence > 0.90:
                            st.info("This text appears to be safe and non-harmful.")
                        else:
                            st.info("This text appears to be generally safe, but maintain awareness.")
                    
            else:
                st.warning("Please enter some text to analyze.")
        
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