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
