from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-pro')

# Initialize Flask app
app = Flask(__name__)

# Serve frontend
@app.route('/')
def index():
    return render_template('chat.html')

# Direct Gemini API endpoint
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        response = model.generate_content(
            f"""You are a medical assistant. Answer this question concisely:
            {question}
            
            If the question is not health-related, say "I can only answer medical questions".
            Never provide medical advice - always recommend consulting a doctor."""
        )
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
