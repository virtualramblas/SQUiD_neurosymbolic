import argparse
from data_generator import generate_mock_data_ollama
from ollama_client import OllamaChatClient
from prompts import get_dataset_generation_system_prompt

def main(input_text: str, model_id: str):
    client = OllamaChatClient(
        model=model_id,
        system_prompt=get_dataset_generation_system_prompt()
    )

    print(generate_mock_data_ollama(client, input_text))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_text", type=str, help="The text where to start generation.")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="The Ollama ID of the model to use.")
    args = parser.parse_args()

    main(args.input_text)