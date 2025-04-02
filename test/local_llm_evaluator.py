import os
import argparse
import pandas as pd
import time
from vllm import LLM, SamplingParams
import yaml


 def parse_score(text):
    try:
        answer_part = text.split('<answer>')[-1].split('</answer>')[0].strip()
        return int(answer_part) if answer_part in {'0','1'} else -1
    except:
        return -1

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_flat_dicts(base, override):
    """Shallow merge: override base keys with override keys."""
    return {**base, **override}

def load_with_defaults_flat(config_path):
    config_dir = os.path.dirname(config_path)
    config = load_yaml(config_path)

    merged_config = {}

    # Process defaults
    if "defaults" in config:
        for default_path in config["defaults"]:
            full_path = os.path.join(config_dir, default_path)
            default_config = load_yaml(full_path)
            merged_config = merge_flat_dicts(merged_config, default_config)
        del config["defaults"]

    # Merge the main config last
    final_config = merge_flat_dicts(merged_config, config)
    return final_config

def ArgParser():
    parser = argparse.ArgumentParser(
        description="Load flat YAML config with optional defaults merging"
    )
    parser.add_argument("config_path", type=str, help="Path to YAML config file")

    args = parser.parse_args()
    config = load_with_defaults_flat(args.config_path)
    
    return argparse.Namespace(**config)


def process_csv(input_csv, groundtruth_column, output_csv, model_dir):
    print("PROCESS CSV")
    df = pd.read_csv(input_csv)

    if groundtruth_column not in df.columns or 'generated' not in df.columns:
        raise ValueError(
            f"CSV must contain {groundtruth_column} and 'generated' columns")

    df['prompt'] = df.apply(
        lambda row: f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI accuracy evaluator. Analyze the question, reference answer, and AI's answer. 
Provide your reasoning in <thinking> tags, then give the score (0/1) in <answer> tags. Example:
<thinking>Analysis...</thinking>
<answer>1</answer><|eot_id|><|start_header_id|>user<|end_header_id|>

Question:
{row['Text'].replace('<|begin_of_text|>', '').replace('Answer:', '')}

Reference Answer:
{row[groundtruth_column]}

AI's Answer:
{row['generated']}

Evaluate the AI's answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",  # No leading whitespace
        axis=1
    )
    input_data = df['prompt'].tolist()
    llm = LLM(model=model_dir,  dtype="float16", max_model_len=3472)  # enable_prefix_caching=False, enable_chunked_prefill=Fals for v100
    sampling_params = SamplingParams(temperature=0.1, max_tokens=2048, top_p=0.95, repetition_penalty=1.15, stop=['</answer>'])
    generated_outputs = llm.generate(input_data, sampling_params)
    # Convert outputs to scores, setting invalid responses to -1
    df['result'] = [output.outputs[0].text.strip() for output in generated_outputs]
    df['score'] = df['result'].apply(parse_score)
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
    
    args = ArgParser()
    print("STARTED EVAL")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    process_csv(args.input_csv, args.groundtruth_column,
                args.output_csv, args.model_dir)


if __name__ == "__main__":
    print("CALLING MAIN")
    main()
