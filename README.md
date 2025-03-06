# Medical RAG Chatbot

A Retrieval Augmented Generation chatbot for medical inquiries using Pinecone and Gemini.

## Setup

1. Clone repo:
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=medical-index
GEMINI_API_KEY=your_gemini_key
```

## Usage
```python
from medical_chatbot import handle_user_query

response = handle_user_query("What is diabetes?")
print(response)
```

3. Create requirements:
```text:requirements.txt
python-dotenv==1.0.0
google-generativeai>=0.3.0
pinecone-client>=3.0.0
openai>=1.0.0
requests>=2.28.0
```

4. Create a data ingestion script (optional):
```python:data_ingest.py
import os
import pinecone
from medical_chatbot import get_embeddings

def ingest_data():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    # Sample medical data
    documents = [
        {"id": "1", "text": "Acne is a skin condition..."},
        {"id": "2", "text": "Diabetes is a metabolic disorder..."}
    ]
    
    vectors = []
    for doc in documents:
        vectors.append({
            "id": doc["id"],
            "values": get_embeddings(doc["text"]),
            "metadata": {"text": doc["text"]}
        })
    
    index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
    index.upsert(vectors=vectors)
```

5. Create proper directory structure:

## Running the API

Start the FastAPI server:
```bash
python app.py
```

The API will be available at http://localhost:8000

## API Endpoints

- `POST /ask`: Ask a medical question
  - Request body: `{"question": "What are the symptoms of diabetes?"}`
  - Response: `{"answer": "The symptoms of diabetes include..."}`

- `GET /health`: Health check endpoint
  - Response: `{"status": "healthy"}`

## Interactive Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc