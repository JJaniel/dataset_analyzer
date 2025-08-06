import os
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI # For NVIDIA
from groq import Groq # For Groq
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI # For NVIDIA
from groq import Groq # For Groq

load_dotenv()

def get_llm_response(prompt_template, input_variables, providers_to_try=None):
    """
    Attempts to get a response from an LLM, with fallback mechanisms.
    """
    llm_providers = providers_to_try if providers_to_try is not None else ["google", "nvidia", "groq"]

    for provider in llm_providers:
        if provider == "google":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                print("Google API Key not found. Skipping Google LLM.")
                continue
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                chain = prompt_template | llm
                result = chain.invoke(input_variables)
                return result.content
            except Exception as e:
                print(f"Google LLM failed: {e}. Trying next provider.")
                continue

        elif provider == "nvidia":
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not nvidia_api_key:
                print("NVIDIA API Key not found. Skipping NVIDIA LLM.")
                continue
            try:
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
                    max_tokens=1024,
                    stream=False
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"NVIDIA LLM failed: {e}. Trying next provider.")
                continue

        elif provider == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                print("Groq API Key not found. Skipping Groq LLM.")
                continue
            try:
                client = Groq(api_key=groq_api_key)
                formatted_prompt = prompt_template.format(**input_variables)

                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Groq LLM failed: {e}. Trying next provider.")
                continue
    
    raise Exception("All LLM providers failed.")
