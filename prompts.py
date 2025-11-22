def get_dataset_generation_system_prompt():
    return """
    You are a creative AI that rephrases given sentences into engaging, conversational stories while incorporating all provided datapoints.- Ensure that no information is omitted or added, and skip any datapoints labeled as 'nan'.- Do not rephrase the object of a sentence. For example, if the sentence is 'start date is $9/22/2023 $', do not change the date to a different format.- Respond only with the rephrased sentence without any additional commentary.
    """

def get_dataset_generation_user_prompt_template(sentence):
    return f"""
    Rephrase the following sentence into a conversational story, ensuring all data points are included while skipping 'nan' values. Do not introduce any extra or false details. Original sentence: {sentence} Creative sentence:
    """