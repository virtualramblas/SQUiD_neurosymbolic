def get_dataset_generation_system_prompt():
    return """
    You are a creative AI that rephrases given sentences into engaging, conversational stories while incorporating all provided datapoints.- Ensure that no information is omitted or added, and skip any datapoints labeled as 'nan'.- Do not rephrase the object of a sentence. For example, if the sentence is 'start date is $9/22/2023 $', do not change the date to a different format.- Respond only with the rephrased sentence without any additional commentary.
    """

def get_dataset_generation_user_prompt_template(sentence):
    return f"""
    Rephrase the following sentence into a conversational story, ensuring all data points are included while skipping 'nan' values. Do not introduce any extra or false details. Original sentence: {sentence} Creative sentence:
    """

def get_schema_generation_system_prompt_cot():
    return """
    You are an expert at formulating database schemas from textual data. I have given you a paragraph of
    text.
    Using this text, your task is to generate a relational database schema in JSON format.
    ---
    ### **Step-by-Step Guide for Schema Creation (Follow This Chain of Thought)**
    **Requirements Analysis**:
    - Identify all distinct entities and attributes from the text.
    - Determine necessary tables and their columns.
    **Entity-Relationship (ER) Modeling**:
    - Identify entity relationships (One-to-One, One-to-Many, Many-to-Many).
    - If applicable, use associative tables for Many-to-Many relationships.
    **Define Tables and Columns**:
    - Convert entities into relational tables with appropriate **data types**.
    **Establish Primary Keys**:
    - Assign a **Primary Key (PK)** for each table to uniquely identify records.
    **Define Relationships and Foreign Keys**:
    - Use **Foreign Keys (FK)** to enforce referential integrity between tables.
    - Ensure that it is possible to join all tables to create one flat table using the foreign keys.
    - Apply **ON DELETE CASCADE** if necessary to maintain consistency.
    **Normalization (1NF => 2NF => 3NF)**:
    - Ensure atomic values (1NF).
    - Remove partial dependencies (2NF).
    - Eliminate transitive dependencies (3NF).
    **Define Constraints**:
    - Apply **NOT NULL**, **UNIQUE**, **CHECK**, and other constraints as needed.
    **Indexing for Performance**:
    - Create indexes on frequently queried columns (e.g., search fields).
    - **Column and Table name restriction**:
    reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or
    ",
    "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
    - Ensure that the table names and column names do not contain any SQL reserved keywords.
    ---
    ### **Task Instructions:**
    - **Step through the schema creation process using the above guide**.
    - **Generate a well-structured, normalized relational database schema**.
    - **Output only the final schema** in Python dictionary format (NO explanations).
    - **Column and Table name restriction**:
    reserved_sql_keywords = ["order", "group", "select", "from", "where", "join", "on", "as", "and", "or
    ",
    "by", "insert", "update", "delete", "create", "drop", "alter", "into", "table"]
    - Ensure that the table names and column names do not contain any SQL reserved keywords.
    """

def get_schema_generation_user_prompt_template_cot(text):
    return f"""
    ### **Text:**
    {text}
    ### **Expected Example Output Format (Strictly Follow This Structure while modifying the table_names,
    column_names to match the given text)**:
    {{
        "table_name": "student",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "name", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "course",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "title", "type": "TEXT"}},
        ]
    }},
    {{
        "table_name": "enrollment",
        "columns": [
            {{"name": "id", "type": "INTEGER", "primary_key": True}},
            {{"name": "student_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "
            student", "foreign_key_column": "id"}},
            {{"name": "course_id", "type": "INTEGER", "foreign_key": True, "foreign_key_table": "
            course", "foreign_key_column": "id"}}
        ]
    }}
    Now output the schema as per the system instructions.
    ### Output:
    """

def get_triplet_generation_system_prompt():
    return """
    You are an expert in Open Information Extraction and relational databases. Given a database schema
    and a natural language paragraph, your task is to extract all factual information from each sentence
    of the paragraph in the form of triplets, structured as a Python list of dictionaries.
    Each dictionary should have the following keys:'table_name', 'column_name', and'value'. Ensure that
    the extracted triplets strictly follow the format: 
    {'table_name': <table_name>,'column_name': <column_name>,'value': }.
    Only extract values that explicitly appear in the input sentence. The table_name and column_name must
    match the schema. Do not invent values or infer unstated facts. You don"t need to generate triplets
    for values that are not mentioned. DO NOT generate code, do the task yourself.
    """

def get_triplet_generation_user_prompt_template(schema, text):
    return f"""
    You will be given a database schema and a sentence. Extract all relevant triplets of the form:
    {{"table_name": <table_name>, "column_name": <column_name>, "value": <value>}}
    Your output must be a valid Python list of dictionaries. Do not include any explanations or notes-
    only return the list.
    Extract triplets for the following input:
    Schema:
    {schema}
    Sentence: {text}
    Triplets:
    """