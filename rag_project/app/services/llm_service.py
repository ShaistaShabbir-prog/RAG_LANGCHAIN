from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import time
import huggingface_hub
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import os
from app.config import HF_MODEL, HF_API_TOKEN


class LLMService:
    def __init__(self, model_name, api_token):
        self.model_name = model_name
        self.api_token = api_token
        self.client = InferenceClient(token=self.api_token)  


    def generate_answer(self, query, context):
        full_prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        try:
            response = self.client.text_generation(
                model=self.model_name,  
                prompt=full_prompt,
                max_new_tokens=250,
            )

            # Handle response format properly
            if isinstance(response, str):
                print("Raw response from model:", response)
                return response.strip()  # Directly return the string response

            if isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"].strip()
            else:
                print("Unexpected response format:", response)
                return "Error: Unexpected response format from LLM."
        except Exception as e:
            print(f"Error in generating answer: {e}")
            return "Error: Failed to generate an answer."


# Initialize LLMService with values from .env
llm_service = LLMService(HF_MODEL, HF_API_TOKEN)
