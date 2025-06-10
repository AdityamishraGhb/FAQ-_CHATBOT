# 🤖 Smart FAQ Chatbot API

A FastAPI-based chatbot backend integrated with HuggingFace LLM for answering business FAQs using natural language.  
Perfect for businesses aiming to automate customer support!

---

## 🚀 Features

- ✅ FastAPI Backend
- ✅ HuggingFace LLM Integration (GPT-2 or any available model)
- ✅ Smart prompt formatting and logging
- ✅ SQLite DB for storing query logs and training data
- ✅ Ready for frontend integration (API based)
- ✅ /reset endpoint to reset chatbot state
- ✅ /ask endpoint for querying FAQs

---

## 📁 Project Structure

FAQ-_CHATBOT/ ├── main.py              # FastAPI app entry point ├── chains.py            # Core chatbot logic and prompt handling ├── models.py            # Pydantic models for request/response ├── database.py          # SQLite DB setup and connection ├── test_chatbot.py      # Simple test script for API ├── requirements.txt     # All required Python packages └── README.md            # This file!

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AdityamishraGhb/FAQ-_CHATBOT.git
cd FAQ-_CHATBOT

2. Create and Activate a Virtual Environment (Optional but Recommended)

# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the FastAPI Server

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

You’ll see something like:

Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


---

🧪 Test the API Endpoints

1. Using Swagger UI

Visit http://127.0.0.1:8000/docs

/ask → Submit a question

/reset → Reset the chatbot state


2. Using curl (for terminal testing)

curl -X POST http://127.0.0.1:8000/ask \
-H "Content-Type: application/json" \
-d '{"question": "What are your business hours?"}'


---

🧠 How It Works

User sends a question via /ask

The prompt is formatted and sent to HuggingFace API (Free LLM)

The response is returned to the user and stored in the SQLite DB for training/analytics



---

🔄 Resetting Chatbot State

Send a POST request to:

http://127.0.0.1:8000/reset

This clears the in-memory context (useful in conversations).


---

⚙ Configuration & Customization

You can switch to another LLM by updating the InferenceClient in chains.py

Add your own dataset or rules to enhance chatbot performance



---

📸 Screenshots (Optional)

Add screenshots of Swagger UI, Terminal response, or DB Viewer here.


---

💡 Future Improvements

Add authentication

Frontend integration (React/Vue)

Model fine-tuning with stored training data

Add webhook support for production usage



---

LLM Powered by HuggingFace 🤗
_____________________________