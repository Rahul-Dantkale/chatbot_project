# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from Data.conversation_data import conversation as conv_data
from Data.para_detail_data import para_data

# Load the pretrained model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Rule-based knowledge corpus
rule_based_dict = {q.lower(): a for q, a in conv_data}

# Load preprocessed paragraph corpus
with open("Data/vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("Data/tfidf_matrix.pkl", "rb") as tfidf_file:
    tfidf_matrix = pickle.load(tfidf_file)


# Load the original paragraph corpus
paragraph_corpus = para_data

# Function to generate chatbot response
def generate_response(user_query, conversation_history=""):
    # Normalize user query
    normalized_query = user_query.lower().strip()
    
    # Step 1: Rule-based matching
    if normalized_query in rule_based_dict:
        return rule_based_dict[normalized_query]

    # Step 2: Search the paragraph corpus for open-ended queries
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()

    # Confidence threshold to ensure relevant matches
    if similarity_scores[0, best_match_idx] > 0.2:  # Adjust threshold as needed
        return paragraph_corpus[best_match_idx][1]  # Return matched paragraph

    # Step 3: Fallback to conversational model
    # Truncate conversation history to fit within model limits
    max_input_length = 500  # Ensure the combined input stays within limits
    truncated_history = conversation_history[-max_input_length:]

    context = truncated_history + f"User: {user_query}\nChatbot:"
    inputs = tokenizer.encode(context, return_tensors="pt", padding=True, truncation=True)
    attention_mask = (inputs != tokenizer.pad_token_id).long()  # Create attention mask
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=100,  # Limit the number of tokens generated
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the chatbot's reply
    chatbot_reply = response.split("User:")[-1].split("Chatbot:")[-1].strip()
    return chatbot_reply

'''
use this to run in terminal / vscode

# Main chatbot interaction
print("Chatbot: Hello! How can I assist you today? Type 'exit' to end the chat.")
conversation_history = ""
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = generate_response(user_query, conversation_history)
    conversation_history += f"User: {user_query}\nChatbot: {response}\n"
    print(f"Chatbot: {response}")
'''