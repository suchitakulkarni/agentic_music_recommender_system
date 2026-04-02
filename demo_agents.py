"""Demo script showcasing all agentic features."""
import sys
from pathlib import Path

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src import config


def _test_connection() -> bool:
    """Test LLM backend connection before launching any agent."""
    if config.USE_OPENAI:
        try:
            from src.agents.openai_client import OpenAIClient
            client = OpenAIClient()
            # Minimal call to verify API key and connectivity.
            client.chat_interactive(
                user_message="ping",
                system_prompt="Reply with one word: ready"
            )
            print("OpenAI connection: OK")
            return True
        except Exception as e:
            print(f"OpenAI connection failed: {e}")
            print("Check your OPENAI_API_KEY environment variable.")
            return False
    else:
        try:
            from src.agents.ollama_client import test_ollama_connection
            return test_ollama_connection()
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False


def main_menu():
    """Display main menu and handle selection."""
    print("\n" + "=" * 80)
    print("TAYLOR SWIFT AGENTIC ANALYSIS DEMO")
    print("=" * 80)
    backend = "OpenAI" if config.USE_OPENAI else "Ollama"
    print(f"Backend: {backend} | Model: {config.MODEL}")

    print("\nTesting connection...")
    if not _test_connection():
        return

    while True:
        print("\n" + "-" * 80)
        print("SELECT AGENT:")
        print("-" * 80)
        print("1. Conversational Analysis Assistant")
        print("   Chain-of-thought reasoning + dynamic data retrieval")
        print()
        print("2. Recommendation Agent")
        print("   Explainable recommendations + preference learning")
        print()
        print("3. Multi-Agent Song Analysis (needs debugging)")
        print("   Lyrical, musical, and contextual agents in debate")
        print()
        print("4. Tool-Using Agent (needs debugging)")
        print("   Dynamic tool selection + self-correction")
        print()
        print("5. Memory-Enhanced Agent (needs debugging)")
        print("   Persistent memory + user profiling")
        print()
        print("0. Exit")
        print("-" * 80)

        choice = input("\nSelect (0-5): ").strip()

        if choice == '0':
            print("\nGoodbye!")
            break

        elif choice == '1':
            from src.agents.analysis_assistant import interactive_session
            interactive_session()

        elif choice == '2':
            from src.agents.recommendation_agent import interactive_autonomous_recommendations
            interactive_autonomous_recommendations()

        elif choice == '3':
            from src.agents.multi_agent_system import interactive_autonomous_multi_agent
            interactive_autonomous_multi_agent()

        elif choice == '4':
            from src.agents.tool_agent import interactive_autonomous_agent
            interactive_autonomous_agent()

        elif choice == '5':
            from src.agents.memory_agent import interactive_autonomous_memory
            interactive_autonomous_memory()

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
