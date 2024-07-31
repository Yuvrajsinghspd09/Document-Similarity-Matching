import os
import re
import numpy as np
import pdf2image
import pandas as pd
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import pytesseract
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model for NER
nlp = spacy.load("de_core_news_sm")

# Directories for train and test
TRAIN_DIR = 'train'
TEST_DIR = 'test'
STOPWORDS = set(stopwords.words('german'))

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Append text from each page
    return text

# Preprocess the extracted text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)
    words = [word for word in words if word not in STOPWORDS]  # Remove stopwords
    return ' '.join(words)

# Extract named entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(f"{ent.label_}: {ent.text}")  # Append entity label and text
    return ' '.join(entities)

# Extract structural features from PDF
def extract_structural_features(pdf_path):
    images = convert_from_path(pdf_path)
    structural_features = ""
    for img in images:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        edges = cv2.Canny(gray, 50, 150)  # Detect edges
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:
            structural_features += f"lines: {len(lines)} "  # Count the lines
    return structural_features

# Get invoice texts from directory
def get_invoice_texts(directory):
    invoice_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):  # Check if the file is a PDF
            file_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(file_path)
            preprocessed_text = preprocess_text(text)
            entities = extract_entities(preprocessed_text)
            structural_features = extract_structural_features(file_path)
            combined_features = f"{preprocessed_text} {entities} {structural_features}"
            invoice_texts[filename] = combined_features
    return invoice_texts

# Calculate similarity between texts
def calculate_similarity(train_texts, test_text):
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_texts.values())
    test_vector = vectorizer.transform([test_text])
    similarities = cosine_similarity(test_vector, train_vectors).flatten()
    return similarities

# Find the most similar invoice
def find_most_similar_invoice(train_texts, test_texts):
    results = {}
    for test_filename, test_text in test_texts.items():
        similarities = calculate_similarity(train_texts, test_text)
        most_similar_index = np.argmax(similarities)
        most_similar_invoice = list(train_texts.keys())[most_similar_index]
        similarity_score = similarities[most_similar_index]
        results[test_filename] = (most_similar_invoice, similarity_score)
    return results

# Main function
def main():
    train_texts = get_invoice_texts(TRAIN_DIR)
    test_texts = get_invoice_texts(TEST_DIR)
    
    results = find_most_similar_invoice(train_texts, test_texts)
    
    # Print results
    for test_invoice, (matched_invoice, score) in results.items():
        print(f"Test Invoice: {test_invoice}")
        print(f"Matched Invoice: {matched_invoice}")
        print(f"Similarity Score: {score:.4f}")
        print()

if __name__ == "__main__":
    main()  # Call main function
