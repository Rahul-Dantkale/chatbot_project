# chatbot_project
This is a chatbot project based on NLP and AI
Chatbot_Project/
├── app.py                 # Flask application backend to serve the chatbot and handle routes.
├── test2.py               # Core chatbot logic, integrating rule-based, retrieval-based, and fallback mechanisms.
├── data_process.py        # Preprocessing script to generate and save the TF-IDF vectorizer and matrix.
├── requirements.txt       # List of Python dependencies required for the project (Flask, transformers, scikit-learn, etc.).
├── static/                # Directory for static files such as CSS, images, and JavaScript used in Flask.
│   └── styles.css         # CSS file for styling the chatbot UI when served through Flask.
├── templates/             # Directory for Flask templates (HTML files served by Flask).
│   └── chatbotUI1.html    # Chatbot UI HTML template integrated with Flask's backend.
├── Data/                  # Directory for chatbot datasets and preprocessed files.
│   ├── conversation_data.py   # Rule-based Q&A dataset for predefined chatbot responses.
│   ├── para_detail_data.py    # Paragraph-based corpus for retrieval-based responses.
│   ├── vectorizer.pkl         # Pickle file storing the pre-trained TF-IDF vectorizer for efficient text similarity.
│   └── tfidf_matrix.pkl       # Pickle file storing the TF-IDF matrix for retrieval-based text matching.
