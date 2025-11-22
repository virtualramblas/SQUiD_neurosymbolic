import spacy

def extract_symbolic_triplets(text):
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


#TODO Use a LLM to generate schema-aligned triplets, in the form (table, column, value)
