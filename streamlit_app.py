import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy
import re
from PyPDF2 import PdfReader
from transformers import pipeline

def extract_and_clean_text(pdf_path, start_page):

    extracted_text = ""

    reader = PdfReader(pdf_path)

    total_pages = len(reader.pages)
    for page_number in range(start_page - 1, total_pages):
        page = reader.pages[page_number]
        page_text = page.extract_text()
        cleaned_text = ""
        cleaned_text = re.sub(r"<<.*?>>", "", page_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"p\. \d+", "", cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"Page \d+|Act [IVX]+|Scene \d+", "", cleaned_text)
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        extracted_text += cleaned_text

    return extracted_text

def extract_title(text_list, name="William Shakespeare"):
    results = []
    for text in text_list:
        match = re.search(r"(\d+)\s+(.*?)\s+" + re.escape(name), text)
        if match:
          text = match.group(2)
          match = re.search(r".*\d+\s+(.*)", text)
          pattern = re.compile(rf"by$", re.IGNORECASE)
          if match:
            cleaned_text = re.sub(pattern, "", match.group(1)).strip()
            results.append(cleaned_text)
          else:
            cleaned_text = re.sub(pattern, "", text).strip()
            results.append(cleaned_text)

        else:
            results.append(None)

    return results

def tokenize_large_text(text, chunk_size=100000):

    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    tokens = []
    for chunk in chunks:
        doc = nlp(chunk)
        chunk_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        tokens.extend(chunk_tokens)
    
    return tokens

pdf_path = "Shakespeare-Complete-Works.pdf"
pdf_text = extract_and_clean_text(pdf_path,77)
plays = pdf_text.split("THE END")

titles = extract_title(plays)
titles = pd.DataFrame(titles, columns=['Title']).head()

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 4703297

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def main():
    st.title("Question Answering App")

    title = st.selectbox("Select a Play:", titles["Title"])
    index = titles.index[titles['Title'] == title].tolist()[0]
    tokens = tokenize_large_text(plays[index])
    copus = " ".join(tokens)

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question.strip():
            answer = qa_pipeline(question=question, context=copus)
            st.write("**Answer:**", answer["answer"])
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
