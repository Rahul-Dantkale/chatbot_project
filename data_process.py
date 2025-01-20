import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from Data.para_detail_data import para_data  # Ensure data.py contains your paragraph corpus

# Extract paragraphs for vectorization
paragraph_texts = [paragraph for _, paragraph in para_data]

# Generate the TF-IDF vectorizer and matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(paragraph_texts)

# Save the vectorizer and TF-IDF matrix for later use
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

with open("tfidf_matrix.pkl", "wb") as tfidf_file:
    pickle.dump(tfidf_matrix, tfidf_file)

print("Vectorizer and TF-IDF matrix saved successfully.")
