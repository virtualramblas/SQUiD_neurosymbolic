import argparse
from data_generator import generate_mock_data_ollama
from ollama_client import OllamaChatClient
from prompts import (get_dataset_generation_system_prompt, 
                    get_schema_generation_system_prompt_cot,
                    get_triplet_generation_system_prompt
)
from schema_generator import generate_schema_ollama
from value_identification import extract_symbolic_triplets, generate_schema_aligned_triplets_ollama

def main(input_file_path: str, model_id: str, input_text: str):
    client = OllamaChatClient(
        model=model_id,
        system_prompt=get_dataset_generation_system_prompt()
    )

    
    if input_file_path != '':
        ### Read text from file ###
        input_file = open(input_file_path)
        input_content = input_file.readlines()
    else:
        if input_text != '':
            ### Generate mock text data ###
            generated_mock_text = generate_mock_data_ollama(client, input_text)
            print(generated_mock_text)
            input_content = [generated_mock_text]

    ### Generate database schema ###
    client.reset()
    client.add_assistant_message(get_schema_generation_system_prompt_cot())
    generated_schema = generate_schema_ollama(client, input_content)
    print(generated_schema)


    ### Generate symbolic triplets ###
    symbolic_triplet_list = []
    for paragraph in input_content:
        symbolic_triplets = extract_symbolic_triplets(paragraph)
        for triplet in symbolic_triplets:
            print(triplet)
            symbolic_triplet_list.append(triplet)

    ### Generate schema-aligned tryplets ###
    client.reset()
    client.add_assistant_message(get_triplet_generation_system_prompt())
    schema_aligned_triplet_list = []
    for paragraph in input_content:
        schema_aligned_triplets = generate_schema_aligned_triplets_ollama(client, generated_schema, paragraph)
        print(schema_aligned_triplets)
        schema_aligned_triplet_list.append(schema_aligned_triplets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesis of relational databases from unstructured text")
    parser.add_argument("--input_file", type=str, default='', help="The path of the input document.")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="The Ollama ID of the model to use.")
    parser.add_argument("--input_text", type=str, default='', help="The text where to start generation (when no input file is provided).")
    args = parser.parse_args()

    main(args.input_file, args.model, args.input_text)