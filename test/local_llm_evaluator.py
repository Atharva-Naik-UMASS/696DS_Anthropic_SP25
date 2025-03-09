import os
import argparse
import pandas as pd
import time
from vllm import LLM, SamplingParams

def process_csv(input_csv, groundtruth_column, output_csv, model_dir):
    print("PROCESS CSV")
    df = pd.read_csv(input_csv)
    
    if groundtruth_column not in df.columns or 'generated' not in df.columns:
        raise ValueError(f"CSV must contain {groundtruth_column} and 'generated' columns")
    
    df['prompt'] = df.apply(
        lambda row: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Your job is to check whether the AI's answer is correct. Compare it with the correct answer and score it as either 0 if the AI's answer is wrong or 1 if it is correct. DO NOT provide any explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
Correct Answer: {row[groundtruth_column]}
AI Answer: {row['generated']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Score: """,
        axis=1
    )
    input_data = df['prompt'].tolist()
    llm = LLM(model=model_dir,  dtype="float16", enable_prefix_caching=False, enable_chunked_prefill=False) #enable_prefix_caching=False, enable_chunked_prefill=Fals for v100
    sampling_params = SamplingParams(temperature=0.5, max_tokens = 2048)
    generated_outputs = llm.generate(input_data, sampling_params)
    # Convert outputs to scores, setting invalid responses to -1
    df['result'] = [
        output.outputs[0].text.strip() for output in generated_outputs
    ]  
    df['score'] = [
        int(output.outputs[0].text.strip())  # Try to convert to integer
        if output.outputs[0].text.strip().isdigit()  # Check if it's a valid integer string
        else -1  # Set invalid responses to -1
        for output in generated_outputs
    ]    
    df.to_csv(output_csv, index=False)
    # Calculate the total score, excluding invalid responses (-1)
    valid_scores = df[df['score'] != -1]['score']  # Filter out -1
    total_score = valid_scores.sum()  # Sum only valid scores
    total_rows = len(valid_scores)  # Count only valid responses

    # Print results
    print(f"Total Score: {total_score}/{total_rows}")
    print(f"Accuracy: {(total_score/total_rows)*100:.2f}%")
    print(f"\nResults saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated responses against reference responses")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model is stored")
    parser.add_argument("--groundtruth_column", type=str, required=True, help="Column name of ground truth")
    parser.add_argument("--output_csv", type=str, default="scored_results.csv", help="Path to output CSV file")
    args = parser.parse_args()
    print("STARTED EVAL", args.input_csv, args.groundtruth_column, args.output_csv)

    process_csv(args.input_csv, args.groundtruth_column, args.output_csv, args.model_dir)

if __name__ == "__main__":
    print("CALLING MAIN")
    main()