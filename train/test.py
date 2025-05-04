import os
import sys
from transformers import AutoTokenizer
from functools import partial
import pandas as pd

# Import the masking function from your main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_no_label import preprocess_mask_labels

def test_mask_labels():
    # Sample data similar to your example
    data = [{
        "text": "<|begin_of_text|>What is the sentiment of the below review? Provide a rating from 1 to 5 stars, where 1 is very negative and 5 is very positive.\nWow, when I first heard this album, it changed my outlook on music forever! The rocking melodies and lyrics on this album are what make it good. This album rivals Be Here Now with it's greatness. OASIS ARE THE BEST!!!!!!!\nRating:\n5<|end_of_text|>"
    }]
    
    # Create a simple dataset
    examples = {"text": [item["text"] for item in data]}
    
    # Load tokenizer (use the same one as in your training)
    model_name = "Claude-3-Opus-20240229"  # Replace with your actual model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        # Fallback to a common tokenizer if the specific one isn't available
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Process the example
    preprocess_fn = partial(preprocess_mask_labels, tokenizer=tokenizer, text_column="text", max_seq_length=512)
    result = preprocess_fn(examples)
    
    # Get the input tokens and labels
    input_ids = result["input_ids"][0]
    labels = result["labels"][0]
    
    # Decode the tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Print results for visual inspection
    print("Input Text:")
    print(examples["text"][0])
    print("\nToken by Token Analysis:")
    print(f"{'Index':<7}{'Token':<20}{'Token ID':<10}{'Label':<10}")
    print("-" * 50)
    
    # Find the label span in the original text
    text = examples["text"][0]
    end_pos = text.find("<|end_of_text|>")
    last_newline = text.rfind("\n", 0, end_pos)
    label_start = last_newline + 1 if last_newline != -1 else 0
    label_end = end_pos
    label_content = text[label_start:label_end]
    
    # Print token analysis
    has_masked_tokens = False
    for i, (token, token_id, label) in enumerate(zip(tokens, input_ids, labels)):
        is_masked = label == -100
        if is_masked:
            has_masked_tokens = True
        print(f"{i:<7}{token:<20}{token_id:<10}{label if not is_masked else 'MASKED':<10}")
    
    # Print summary
    print("\nSummary:")
    print(f"Label content that should be masked: '{label_content}'")
    print(f"Found masked tokens: {has_masked_tokens}")
    
    # Verify if any non-label tokens are masked or if any label tokens are not masked
    decoded_input = tokenizer.decode(input_ids)
    masked_text = ""
    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        if label == -100:
            token = tokenizer.decode([token_id])
            masked_text += token
    
    print(f"\nMasked text: '{masked_text.strip()}'")
    print(f"Expected masked text (approximate): '{label_content.strip()}'")
    
    success = label_content.strip() in masked_text.strip() or masked_text.strip() in label_content.strip()
    print(f"\nTest {'PASSED' if success and has_masked_tokens else 'FAILED'}")

if __name__ == "__main__":
    test_mask_labels()