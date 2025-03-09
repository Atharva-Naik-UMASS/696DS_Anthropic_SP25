import os
import argparse
import pandas as pd
import time
from vllm import LLM, SamplingParams

def process_csv(input_csv, output_csv, model_dir):
    print("PROCESS CSV")
    df = pd.read_csv(input_csv)
    
    if 'response' not in df.columns or 'generated' not in df.columns:
        raise ValueError("CSV must contain 'response' and 'generated' columns")
    
    df['prompt'] = df.apply(
        lambda row: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Your job is to check whether the AI's answer is correct. Compare it with the correct answer and score it as either 0 if the AI's answer is wrong or 1 if it is correct. DO NOT provide any explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Correct Answer: {row['response']}
AI Answer: {row['generated']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        axis=1
    )
    input_data = df['prompt'].tolist()
    llm = LLM(model=model_dir,  dtype="float16", enable_prefix_caching=False, enable_chunked_prefill=False) #enable_prefix_caching=False, enable_chunked_prefill=Fals for v100
    sampling_params = SamplingParams(temperature=0.5, max_tokens = 2048)
    generated_outputs = llm.generate(input_data, sampling_params)
    df['score'] = [
        int(output.outputs[0].text.strip())  # Try to convert to integer
        if output.outputs[0].text.strip().isdigit()  # Check if it's a valid integer string
        else 0  # Default to 0 if conversion fails
        for output in generated_outputs
    ]    
    df.to_csv(output_csv, index=False)
    # df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
    # Count the number of 1s in the 'score' column
    total_score = df['score'].sum()

    # Calculate accuracy (percentage of 1s)
    total_rows = len(df)
    accuracy = (total_score / total_rows) * 100

    # Print results
    print(f"Total Score: {total_score}/{total_rows}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nResults saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated responses against reference responses")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model is stored")
    parser.add_argument("--output_csv", type=str, default="scored_results.csv", help="Path to output CSV file")
    args = parser.parse_args()
    print("STARTED EVAL", args.input_csv, args.output_csv)

    process_csv(args.input_csv, args.output_csv, args.model_dir)

if __name__ == "__main__":
    print("CALLING MAIN")
    main()