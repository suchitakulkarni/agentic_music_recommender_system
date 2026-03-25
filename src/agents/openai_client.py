"""OpenAI API client wrapper with consistent interface matching OllamaClient."""
import os
from openai import OpenAI
import src.config as config

class OpenAIClient:
    """Wrapper for OpenAI API with consistent interface."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
        """
        self.model = model
        self.conversation_history = []
        
        # Get API key from config or environment
        api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in config.py or as environment variable.")
        
        self.client = OpenAI(api_key=api_key)
    
    def chat_interactive(self, user_message: str, system_prompt: str = None, stream: bool = False, max_tokens: int = 500):
        """
        Send a chat message to OpenAI.
        
        Args:
            user_message: The user's question
            system_prompt: Optional system prompt for context
            stream: Whether to stream the response (not implemented)
            
        Returns:
            str: The model's response
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens  # Limit response length for speed
            )
            
            assistant_message = response.choices[0].message.content
            
            # Save to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

def test_openai_connection():
    """Test if OpenAI is running and accessible."""
    try:
        client = OpenAIClient()
        response = response = client.chat_interactive("Say 'hello' if you can hear me.")
        print(f"✓ OpenAI is running!")
        print(f"  Model: {config.MODEL}")
        print(f"  Test response: {response}")
        return True
    except Exception as e:
        print(f"✗ OpenAI connection failed: {e}")
        print("\nMake sure OpenAI is running:")
        return False
