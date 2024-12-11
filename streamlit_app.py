import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy
import re
from PyPDF2 import PdfReader
import spacy_streamlit
import pickle

titles = pd.read_csv('titles.csv')
with open("copus.pkl", "rb") as file:
    full_copus = pickle.load(file)

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def main():
    st.title("Question Answering App")

    title = st.selectbox("Select a Play:", titles["Title"])
    index = titles.index[titles['Title'] == title].tolist()[0]
    context = full_copus[index]

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question.strip():
            answer = qa_pipeline(question=question, context=context)
            st.write("**Answer:**", answer["answer"])
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
