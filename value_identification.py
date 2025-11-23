import spacy
from ollama_client import OllamaChatClient
from prompts import get_triplet_generation_user_prompt_template

def extract_symbolic_triplets(text: str):
    """Extracting Subject–Relation–Object triplets using the spaCy API (English language only).

    Args:
        text: the input corpus of text.

    Returns:
        The list of extracted triplets.
    
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    triplets = []

    for sent in doc.sents:
        subject = None
        relation = None
        obj = None

        for token in sent:
            # Subject
            if "subj" in token.dep_:
                subject = token.text

            # Relation (verb)
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                relation = token.lemma_

            # Object
            if "obj" in token.dep_:
                obj = token.text

        if subject and relation and obj:
            triplets.append((subject, relation, obj))

    return triplets

def generate_schema_aligned_triplets(ollama_client: OllamaChatClient, schema: str, text: str):
    """Generate, using an LLM, schema-aligned triplets, in the form (table, column, value)

    Args:
        ollama_client: an instance of the client to chat with a model hosted in a Ollama server.
        schema: the reference database schema.
        text: the input corpus of text.

    Returns:
        A list of dictionaries (the generated triplets).
    
    """
    return ollama_client.chat(
        get_triplet_generation_user_prompt_template(schema, text))
