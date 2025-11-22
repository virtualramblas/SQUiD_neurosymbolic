import argparse
from data_generator import generate_mock_data_ollama
from ollama_client import OllamaChatClient
from prompts import get_dataset_generation_system_prompt, get_schema_generation_system_prompt_cot
from schema_generator import generate_schema_ollama
from value_identification import extract_symbolic_triplets

def main(input_text: str, model_id: str):
    client = OllamaChatClient(
        model=model_id,
        system_prompt=get_dataset_generation_system_prompt()
    )

    ### Generate mock text data ###
    generated_mock_text = generate_mock_data_ollama(client, input_text)
    print(generated_mock_text)

    ### Generate database schema ###
    client.reset()
    client.add_assistant_message(get_schema_generation_system_prompt_cot())
    generated_schema = generate_schema_ollama(client, generated_mock_text)
    print(generated_schema)

    ### Generate triplets ###
    symbolic_triplets = extract_symbolic_triplets(generated_mock_text)
    for triplet in symbolic_triplets:
        print(triplet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesis of relational databases from unstructured text")
    parser.add_argument("--input_text", type=str, help="The text where to start generation.")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="The Ollama ID of the model to use.")
    args = parser.parse_args()

    main(args.input_text, args.model)