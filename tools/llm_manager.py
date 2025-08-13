import os
import google.generativeai as genai # Import raw Google GenAI
from openai import OpenAI # For NVIDIA
from groq import Groq # For Groq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_llm_response(prompt_template, input_variables, providers_to_try=None):
    """
    Attempts to get a response from an LLM, with fallback mechanisms.
    Returns the response content and the name of the successful provider.
    """
    llm_providers = providers_to_try if providers_to_try is not None else ["groq", "google", "nvidia", "nvidia_nemotron"] # Prioritize Groq

    for provider in llm_providers:
        print(f"Attempting to use {provider.capitalize()} LLM...")
        try:
            if provider == "google":
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    print("Google API Key not found. Skipping Google LLM.")
                    continue
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                formatted_prompt = prompt_template.format(**input_variables)
                response = model.generate_content(formatted_prompt)
                print(f"Successfully got response from {provider.capitalize()} LLM.")
                return response.text, provider

            elif provider == "nvidia":
                nvidia_api_key = os.getenv("NVIDIA_API_KEY")
                if not nvidia_api_key:
                    print("NVIDIA API Key not found. Skipping NVIDIA LLM.")
                    continue
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=nvidia_api_key
                )
                formatted_prompt = prompt_template.format(**input_variables)

                completion = client.chat.completions.create(
                    model="meta/llama-3.3-70b-instruct",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    stream=False
                )
                print(f"Successfully got response from {provider.capitalize()} LLM.")
                return completion.choices[0].message.content, provider

            elif provider == "nvidia_nemotron":
                nvidia_api_key = os.getenv("NVIDIA_API_KEY")
                if not nvidia_api_key:
                    print("NVIDIA API Key not found. Skipping NVIDIA Nemotron LLM.")
                    continue
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=nvidia_api_key
                )
                formatted_prompt = prompt_template.format(**input_variables)

                completion = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=4096,
                    stream=False
                )
                print(f"Successfully got response from {provider.capitalize()} LLM.")
                return completion.choices[0].message.content, provider

            elif provider == "groq":
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    print("Groq API Key not found. Skipping Groq LLM.")
                    continue
                client = Groq(api_key=groq_api_key)
                formatted_prompt = prompt_template.format(**input_variables)

                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=1,
                    max_completion_tokens=8192,
                    top_p=1,
                    stream=False
                )
                print(f"Successfully got response from {provider.capitalize()} LLM.")
                return completion.choices[0].message.content, provider

        except Exception as e:
            print(f"{provider.capitalize()} LLM failed: {e}. Trying next provider.")
            continue
