from transformers import T5ForConditionalGeneration, T5Tokenizer
import gc
import torch
def t5_summarize(text, model_name='t5-base', max_length=500, min_length=100):
    """
    Summarize the given text using the T5-base model.

    Args:
    text (str): The input text to summarize.
    model_name (str): The model to use, default is 't5-large'.
    max_length (int): The maximum length of the summary.
    min_length (int): The minimum length of the summary.

    Returns:
    str: The summarized text.
    """
    # Set device to CPU
    # device = torch.device("cpu")
    
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Move model to CPU
    # model = model.to(device)
    
    # Preprocess the text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move input tensors to CPU
    # inputs = inputs.to(device)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    del tokenizer
    del model
    gc.collect()

    return summary