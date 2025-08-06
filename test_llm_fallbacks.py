import os
from dotenv import load_dotenv
from tools.llm_manager import get_llm_response
from langchain.prompts import PromptTemplate

load_dotenv()

def test_llm_provider(provider_name, prompt_template, input_variables):
    print(f"\n--- Testing {provider_name} LLM ---")
    try:
        response = get_llm_response(prompt_template, input_variables, providers_to_try=[provider_name])
        print(f"{provider_name} LLM successful! Response snippet: {response[:100]}...")
    except Exception as e:
        print(f"{provider_name} LLM failed: {e}")

if __name__ == "__main__":
    test_prompt_template = "Tell me a very short story about a {animal}."
    test_input_variables = {"animal": "cat"}

    # Note: This test will attempt to use each provider individually.
    # Ensure you have the necessary API keys set in your .env file.

    # Test Google LLM
    test_llm_provider("google", test_prompt_template, test_input_variables)

    # Test NVIDIA LLM
    test_llm_provider("nvidia", test_prompt_template, test_input_variables)

    # Test Groq LLM
    test_llm_provider("groq", test_prompt_template, test_input_variables)