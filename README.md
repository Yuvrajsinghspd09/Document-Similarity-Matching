# Document-Similarity-Matching

# Invoice Similarity Checker

This project is for comparing invoices and finding the most similar one. We use various techniques like text extraction, text preprocessing, named entity recognition (NER), and structural feature extraction.

## Requirements

Before running the code, make sure you have installed the necessary Python packages. You can install them using pip:

```bash
pip install numpy pandas pdf2image PyPDF2 scikit-learn opencv-python pytesseract nltk spacy
python -m spacy download de_core_news_sm
```

How to Use
Training and Test Data:

Place your training invoices in the train directory.
Place your test invoices in the test directory.
Running the Code:

Simply run the script by executing the following command in your terminal:

```python script_name.py```


The script will print the test invoice, the matched invoice from the training set, and the similarity score.
Code Description

Libraries Imported:

os: For file handling.
re: For regular expression operations.
numpy: For numerical operations.
pdf2image: To convert PDF pages to images.
pandas: For handling data.
PyPDF2: For reading PDF files.
scikit-learn: For machine learning tasks.
opencv-python (cv2): For image processing.
pytesseract: For optical character recognition (OCR).
nltk: For text processing.
spacy: For NER.
Functions
extract_text_from_pdf: Extracts text from a PDF file.
preprocess_text: Preprocesses text by removing non-word characters, extra spaces, converting to lowercase, and removing stopwords.
extract_entities: Extracts named entities from the text using spaCy.
extract_structural_features: Extracts structural features from PDF images like the number of lines.
get_invoice_texts: Gets preprocessed text, entities, and structural features from all invoices in a directory.
calculate_similarity: Calculates the cosine similarity between training and test texts.
find_most_similar_invoice: Finds the most similar invoice from the training set for each test invoice.
main: Main function to orchestrate the process.
Main Process
Load Data: Load training and test invoices.
Find Similarity: Calculate similarity between test and training invoices.
Print Results: Print the matched invoice and similarity score for each test invoice.
Notes
Make sure tesseract is installed on your system and is added to the system PATH.
The script uses German language stopwords and NER model. Change them if your invoices are in another language.
