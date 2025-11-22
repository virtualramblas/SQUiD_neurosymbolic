import json
from ollama_client import OllamaChatClient
from prompts import get_schema_generation_user_prompt_template_cot

def generate_schema_ollama(ollama_client: OllamaChatClient, text: str) -> str:
    """Generates databaase schema for the input unstructured text.

    Args:
        ollama_client: an instance of the client to chat with a model hosted in a Ollama server.
        sentence: the input text.

    Returns:
        The generated schema in JSON format.
    
    """
    raw_generated_schema = ollama_client.chat(
        get_schema_generation_user_prompt_template_cot(text))
    try:
        json_string = raw_generated_schema.strip()
        return json_string
    except json.JSONDecodeError as e:
        print(f"\nError decoding JSON: {e}")
        print("\nGenerated output is not a valid JSON structure.")
        return None