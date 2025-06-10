import os
import requests
import json
from typing import Optional, Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from db import get_training_data

# Load environment variables
load_dotenv()

class FreeLLMProvider:
    """Free LLM provider using Hugging Face Inference API"""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN', '')}",
            "Content-Type": "application/json"
        }
        # Fallback models that don't require API tokens
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-small"
        ]
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using Hugging Face free models"""
        
        # Try the primary model first
        response = self._try_model(self.api_url, prompt, max_length)
        if response:
            return response
        
        # Try fallback models
        for model in self.fallback_models:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            response = self._try_model(api_url, prompt, max_length)
            if response:
                return response
        
        # Final fallback - simple rule-based response
        return self._rule_based_response(prompt)
    
    def _try_model(self, api_url: str, prompt: str, max_length: int) -> Optional[str]:
        """Try to get response from a specific model"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(api_url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text'].strip()
                    elif 'response' in result[0]:
                        return result[0]['response'].strip()
            
            return None
            
        except Exception as e:
            print(f"Error with model {api_url}: {str(e)}")
            return None
    
    def _rule_based_response(self, prompt: str) -> str:
        """Simple rule-based responses as ultimate fallback"""
        prompt_lower = prompt.lower()
        
        # Check training data first
        training_data = get_training_data()
        for data in training_data:
            if any(word in prompt_lower for word in data['question'].lower().split()):
                return data['answer']
        
        # Common FAQ patterns
        if any(word in prompt_lower for word in ['return', 'refund', 'policy']):
            return "You can return any item within 7 days of purchase with original receipt. Items must be in original condition."
        
        elif any(word in prompt_lower for word in ['shipping', 'delivery', 'ship']):
            return "We offer free shipping on orders over $50. Standard delivery takes 3-5 business days."
        
        elif any(word in prompt_lower for word in ['payment', 'pay', 'credit', 'card']):
            return "We accept all major credit cards, PayPal, and bank transfers. Payment is processed securely."
        
        elif any(word in prompt_lower for word in ['business', 'hours', 'open', 'time']):
            return "We are open Monday to Friday, 9 AM to 6 PM EST. Customer service is available during business hours."
        
        elif any(word in prompt_lower for word in ['contact', 'support', 'help', 'customer']):
            return "You can contact our customer support at support@company.com or call 1-800-123-4567 during business hours."
        
        elif any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greet']):
            return "Hello! How can I help you today? I'm here to answer your questions about our products and services."
        
        elif any(word in prompt_lower for word in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help you with?"
        
        else:
            return f"Thank you for your question about '{prompt}'. I'd be happy to help you with information about our products, services, returns, shipping, or any other inquiries you may have."

class GoogleLLMProvider:
    """Google Gemini LLM provider (requires API key)"""
    
    def __init__(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not found")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=self.api_key,
                temperature=0.7,
                convert_system_message_to_human=True
            )
            self.available = True
        except Exception as e:
            print(f"Google LLM not available: {str(e)}")
            self.available = False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Google Gemini"""
        if not self.available:
            return "Google LLM is not available. Please check your API key."
        
        try:
            full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:"
            response = self.llm.invoke(full_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"

class FAQChatbot:
    def __init__(self):
        """Initialize the FAQ chatbot with the specified LLM provider."""
        self.llm_provider = os.getenv("LLM_PROVIDER", "huggingface").lower()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
        
        # Initialize LLM provider
        if self.llm_provider == "google":
            self.llm = GoogleLLMProvider()
        else:
            self.llm = FreeLLMProvider()
        
        print(f"✅ FAQ Chatbot initialized with {self.llm_provider} LLM")
    
    def ask(self, question: str, session_id: Optional[str] = None) -> str:
        """Ask a question and get an AI response."""
        try:
            # Get conversation context
            context = self._build_context()
            
            # Generate response based on provider
            if self.llm_provider == "google" and hasattr(self.llm, 'available') and self.llm.available:
                response = self.llm.generate_response(question, context)
            else:
                # Use free LLM with context
                prompt_with_context = f"{context}\nUser: {question}\nAssistant:"
                response = self.llm.generate_response(prompt_with_context)
                
                # Clean up response if it includes the prompt
                if "User:" in response:
                    response = response.split("Assistant:")[-1].strip()
                if "User:" in response:
                    response = response.split("User:")[0].strip()
            
            # Save to memory
            self.memory.save_context(
                {"question": question},
                {"answer": response}
            )
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _build_context(self) -> str:
        """Build context from training data and conversation history"""
        context_parts = []
        
        # Add system instructions
        context_parts.append("You are a helpful customer service assistant. Provide clear, concise, and helpful answers.")
        
        # Add training data as context
        training_data = get_training_data()
        if training_data:
            context_parts.append("\nHere are some example responses:")
            for data in training_data[:5]:  # Limit to recent 5
                context_parts.append(f"Q: {data['question']}\nA: {data['answer']}")
        
        # Add recent conversation history
        messages = self.memory.chat_memory.messages
        if messages:
            context_parts.append("\nRecent conversation:")
            for msg in messages[-6:]:  # Last 3 Q&A pairs
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
    
    def get_memory_summary(self) -> dict:
        """Get a summary of the current conversation memory."""
        messages = self.memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "conversation_pairs": len(messages) // 2,
            "llm_provider": self.llm_provider,
            "memory_buffer": str(self.memory.buffer) if hasattr(self.memory, 'buffer') else "Active"
        }

# Global chatbot instance
chatbot = None

def get_chatbot() -> FAQChatbot:
    """Get or create the global chatbot instance."""
    global chatbot
    if chatbot is None:
        chatbot = FAQChatbot()
    return chatbot

def reset_chatbot():
    """Reset the global chatbot instance."""
    global chatbot
    if chatbot:
        chatbot.clear_memory()
    chatbot = None