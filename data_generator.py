from ollama_client import OllamaChatClient
from prompts import get_dataset_generation_user_prompt_template

def generate_mock_data_ollama(ollama_client: OllamaChatClient, sentence: str) -> str:
    """Generates mock test data to test the SQUiD workflow.

    Args:
        ollama_client: an instance of the client to chat with a model hosted in a Ollama server.
        sentence: the input text.

    Returns:
        The generated text.
    
    """
    return ollama_client.chat(
        get_dataset_generation_user_prompt_template(sentence))
