import os
from dotenv import load_dotenv
from tools.llm_manager import get_llm_response
from langchain.prompts import PromptTemplate

load_dotenv()

if __name__ == "__main__":
    test_prompt_template = "Tell me a very short story about a {animal}."
    test_input_variables = {"animal": "cat"}

    print("\n--- Testing LLM Fallback Mechanism ---")
    print("Attempting to get a response using the defined fallback order (Google -> NVIDIA -> Groq).")
    print("If Google fails, it should automatically try NVIDIA, and then Groq.")

    try:
        response = get_llm_response(test_prompt_template, test_input_variables)
        print("\nFallback successful! Final response snippet:")
        print(f"{response[:200]}...")
    except Exception as e:
        print(f"\nAll LLM providers failed: {e}")
