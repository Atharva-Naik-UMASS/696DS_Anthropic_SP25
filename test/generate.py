from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import argparse
import os
import pandas as pd

def load_test_data(test_csv, input_field):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV file not found at {test_csv}")
    
    df = pd.read_csv(test_csv)
    print("Test data loaded successfully!")
    
    input_data = df[input_field].tolist()
    
    return df, input_data

def main():
    parser = argparse.ArgumentParser(description="Load model and test data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model is stored")
    parser.add_argument("--adapter_dir", type=str, required=False, help="Directory where the adapter is stored")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--target_field", type=str, required=False, help="Target column name (if available)")
    parser.add_argument("--input_field", type=str, required=True, help="Comma-separated list of input feature columns")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="Path to save the output CSV file")

    args = parser.parse_args()
    llm = None
    if args.adapter_dir:
        llm = LLM(model=args.model_dir, enable_lora=True, dtype="float16", max_lora_rank=64, enable_prefix_caching=False, enable_chunked_prefill=False)
    else:
        llm = LLM(model=args.model_dir,  dtype="float16", enable_prefix_caching=False, enable_chunked_prefill=False) #enable_prefix_caching=False, enable_chunked_prefill=Fals for v100
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens = 2048)

    df, input_data = load_test_data(args.test_csv, args.input_field) 
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