from flask import Flask, render_template, request, jsonify
from test2 import generate_response  # Import chatbot logic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chatbotUI1.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    if user_message:
        chatbot_response = generate_response(user_message)  # Use the logic from test2.py
        return jsonify({"response": chatbot_response})
    return jsonify({"response": "Sorry, I didn't understand that."})

if __name__ == "__main__":
    app.run(debug=True)
