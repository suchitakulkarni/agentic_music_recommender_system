"""Ollama client wrapper for consistent API."""
import ollama
import json
from typing import List, Dict, Optional


class OllamaClient:
    """Wrapper for Ollama API with consistent interface."""
    
    def __init__(self, model: str = "llama3.1:8b"):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (llama3.1:8b, mistral:7b, qwen2.5:7b, etc.)
        """
        self.model = model
        self.conversation_history = []



    
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 1024,
                 temperature: float = 0.7,
                 stream: bool = False) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stream: Stream response token by token
            
        Returns:
            Generated text
        """
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of {'role': 'user'/'assistant', 'content': '...'}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant's response
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    def chat_interactive(self, 
                        user_message: str,
                        system_prompt: Optional[str] = None,
                        reset_history: bool = False) -> str:
        """
        Interactive chat with automatic history management.
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt (only used on first message)
            reset_history: Clear conversation history
            
        Returns:
            Assistant's response
        """
        if reset_history:
            self.conversation_history = []
        
        # Add system prompt on first message
        if system_prompt and len(self.conversation_history) == 0:
            self.conversation_history.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add user message
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Get response
        response = self.chat(self.conversation_history)
        
        # Add assistant response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    try:
        client = OllamaClient()
        response = client.generate("Say 'hello' if you can hear me.", max_tokens=20)
        print(f"✓ Ollama is running!")
        print(f"  Model: {client.model}")
        print(f"  Test response: {response}")
        return True
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Start Ollama: 'ollama serve' (in separate terminal)")
        print("  2. Pull a model: 'ollama pull llama3.1:8b'")
        return False


if __name__ == "__main__":
    test_ollama_connection()
