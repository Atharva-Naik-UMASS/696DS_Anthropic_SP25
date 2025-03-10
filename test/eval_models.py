import argparse
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score
import os

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {key: scores[key].fmeasure for key in scores}

def compute_bleu(references, hypothesis):
    references = [[ref.split()] for ref in references]
    hypothesis = [hyp.split() for hyp in hypothesis]
    return corpus_bleu(references, hypothesis)

def compute_length(target, generated):
    return len(target.split()), len(generated.split())

def evaluate(args):
    test_data = pd.read_csv(args.test_csv)
    generated_data = pd.read_csv(args.generated_output_csv)
    
    if args.target_field not in test_data.columns:
        print(f"Error: '{args.target_field}' not found in the test CSV.")
        return
    if args.generated_field not in generated_data.columns:
        print(f"Error: '{args.generated_field}' not found in the generated output CSV.")
        return

    total_scores = {"target" : [] , "generated" : [], "rouge1": [], "rouge2": [], "rougeL": [], "bleu": [], "accuracy": [], "length_target": [], "length_generated": []}
    outputs = []
    for index, row in test_data.iterrows():
        target = row[args.target_field]
        output = generated_data.iloc[index][args.generated_field]
        # total_scores["target"].append(target)
        # total_scores["generated"].append(output)
        
        if 'rouge' in args.tests:
            rouge_scores = compute_rouge(target, output)
            for key in rouge_scores:
                total_scores[key].append(rouge_scores[key])

        if 'bleu' in args.tests:
            total_scores['bleu'].append(compute_bleu([target], [output]))

        if 'accuracy' in args.tests:
            is_correct = int(str(target).strip() == str(output).strip())  # Ensure both are strings and strip whitespace
            total_scores['accuracy'].append(is_correct)

        if 'length' in args.tests:
            target_len, generated_len = compute_length(target, output)
            total_scores['length_target'].append(target_len)
            total_scores['length_generated'].append(generated_len)
    
    # Compute average values 
    avg_scores = {test: sum(values) / len(values) for test, values in total_scores.items() if values}
    print(avg_scores)
    avg_scores["test_csv"] = args.test_csv
    avg_scores["generated_output_csv"] = args.generated_output_csv
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame([avg_scores])
    results_df.to_csv(args.output_csv, index=False)
    print(f"Averaged results saved to {args.output_csv}")
    
    total_scores_df = pd.DataFrame([total_scores])
    total_scores_df.to_csv("total_scores.csv", index=False)
    print(f"Averaged results saved to total_scores.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLP Models on generated outputs")
    parser.add_argument("test_csv", type=str, help="Path to the test CSV (ground truth)")
    parser.add_argument("generated_output_csv", type=str, help="Path to the generated output CSV")
    parser.add_argument("target_field", type=str, help="Target column name in the test CSV")
    parser.add_argument("generated_field", type=str, help="Generated output column name in the generated CSV")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file with scores")
    parser.add_argument("tests", type=str, nargs='+', choices=['rouge', 'bleu', 'accuracy', 'length'], help="List of tests to evaluate")
    
    args = parser.parse_args()
    evaluate(args)
