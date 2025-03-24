from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import argparse
import os
import pandas as pd

import os
import pandas as pd
import random

def load_test_data(test_csv, input_field, train_csv, balance_label, shots=0):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV file not found at {test_csv}")
    
    df = pd.read_csv(test_csv)
    print("Test data loaded successfully!")
    
    input_data = df[input_field].tolist()
    
    # If shots > 0, load training data and sample shots
    if shots > 0:
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train CSV file not found at {train_csv}")
        
        train_df = pd.read_csv(train_csv)
        print("Train data loaded successfully!")
        
        if not balance_label:
            # Randomly sample 'shots' number of rows from the 'Text' field
            sampled_texts = train_df['Text'].sample(n=shots, random_state=42).tolist()
        else:
            # Get unique groups based on balance_label and compute sample sizes
            unique_labels = train_df[balance_label].unique()
            n_groups = len(unique_labels)
            shots_per_group = shots // n_groups
            remainder = shots % n_groups
            
            sampled_texts = []
            for label in unique_labels:
                group_df = train_df[train_df[balance_label] == label]
                n_samples = shots_per_group
                # Distribute any extra samples among the first few groups
                if remainder > 0:
                    n_samples += 1
                    remainder -= 1
                if len(group_df) < n_samples:
                    raise ValueError(f"Not enough rows in group '{label}' to sample {n_samples} shots.")
                group_sample = group_df['Text'].sample(n=n_samples, random_state=42).tolist()
                sampled_texts.extend(group_sample)
        
        # Clean the sampled texts from any markers
        sampled_texts = [text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "") 
                         for text in sampled_texts]
        
        # Prepend the sampled texts to each input data item
        for i in range(len(input_data)):
            input_data[i] = "<|begin_of_text|>" + "\n".join(sampled_texts) + input_data[i].replace("<|begin_of_text|>", "\n")
    return df, input_data

def main():
    parser = argparse.ArgumentParser(description="Load model and test data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model is stored")
    parser.add_argument("--adapter_dir", type=str, required=False, help="Directory where the adapter is stored")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--target_field", type=str, required=False, help="Target column name (if available)")
    parser.add_argument("--input_field", type=str, required=True, help="Comma-separated list of input feature columns")
    parser.add_argument("--shots", type=int,  default=0, help="Number of shots in prompting")
    parser.add_argument("--train_csv", type=str, default="", help="train csv file to get data for prompting")
    parser.add_argument("--balance_label", type=str, default="", help="label to balance the rows we get for n shot prompting")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="Path to save the output CSV file")

    args = parser.parse_args()
    llm = None
    if args.adapter_dir:
        llm = LLM(model=args.model_dir, enable_lora=True, dtype="float16", max_lora_rank=64, enable_prefix_caching=False, enable_chunked_prefill=False)
    else:
        llm = LLM(model=args.model_dir,  dtype="float16", enable_prefix_caching=False, enable_chunked_prefill=False) #enable_prefix_caching=False, enable_chunked_prefill=Fals for v100
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens = 2048)

    df, input_data = load_test_data(args.test_csv, args.input_field, args.train_csv, args.balance_label, args.shots) 
    generated_outputs = None
    if args.adapter_dir:   
        generated_outputs = llm.generate(input_data, sampling_params, lora_request=LoRARequest("test_adapter", 1, args.adapter_dir))
    else:
        generated_outputs = llm.generate(input_data, sampling_params)
    df["generated"] = [output.outputs[0].text for output in generated_outputs]

    df.to_csv(args.output_csv, index=False)
    print(f"Output saved to {args.output_csv}")

    if args.target_field:
        correct = 0
        total = len(df)
        for i, (gen, target) in enumerate(zip(df["generated"], df[args.target_field])):
            if gen.strip() == str(target).strip():
                correct += 1
            else:
                print("Mismatch in Sentence:", df["Sentence"].iloc[i])
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()