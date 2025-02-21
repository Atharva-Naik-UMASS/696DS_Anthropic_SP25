import argparse
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score
import numpy as np

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def compute_bleu(references, hypothesis):
    # BLEU score calculation using NLTK
    references = [[ref.split()] for ref in references]
    hypothesis = [hyp.split() for hyp in hypothesis]
    return corpus_bleu(references, hypothesis)

def compute_accuracy(true_labels, predicted_labels):
    # Accuracy score calculation
    return accuracy_score(true_labels, predicted_labels)

def compute_length(target, generated):
    # Compute word count of both target and generated text
    target_len = len(target.split())
    generated_len = len(generated.split())
    
    # Return both the average target length and average generated length
    return target_len, generated_len

def evaluate(args):
    # Read the input CSV files
    test_data = pd.read_csv(args.test_csv)
    generated_data = pd.read_csv(args.generated_output_csv)

    # Check if the necessary columns exist in both files
    if args.target_field not in test_data.columns:
        print(f"Error: '{args.target_field}' not found in the test CSV.")
        return

    if args.generated_field not in generated_data.columns:
        print(f"Error: '{args.generated_field}' not found in the generated output CSV.")
        return

    # Store the results
    results = []

    # Store total lengths for average computation
    total_generated_length = 0
    total_target_length = 0
    total_rows = len(test_data)

    # Iterate through the dataset assuming they match by index
    for index, row in test_data.iterrows():
        target = row[args.target_field]
        output = generated_data.iloc[index][args.generated_field]

        score_data = {}

        # Evaluate each metric based on the provided tests
        if 'rouge' in args.tests:
            rouge_scores = compute_rouge(target, output)
            score_data['rouge1'] = rouge_scores['rouge1'].fmeasure
            score_data['rouge2'] = rouge_scores['rouge2'].fmeasure
            score_data['rougeL'] = rouge_scores['rougeL'].fmeasure

        if 'bleu' in args.tests:
            bleu_score = compute_bleu([target], [output])
            score_data['bleu'] = bleu_score

        if 'accuracy' in args.tests:
            accuracy = compute_accuracy([target], [output])
            score_data['accuracy'] = accuracy

        if 'length' in args.tests:
            target_len, generated_len = compute_length(target, output)
            score_data['avg_target_length'] = target_len
            score_data['avg_generated_length'] = generated_len

        results.append(score_data)

    # Add average length results to the output file
    results_df = pd.DataFrame(results)

    # Save results to output CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLP Models on generated outputs")

    # Define command-line arguments
    parser.add_argument("test_csv", type=str, help="Path to the test CSV (ground truth)")
    parser.add_argument("generated_output_csv", type=str, help="Path to the generated output CSV")
    parser.add_argument("target_field", type=str, help="Target column name in the test CSV")
    parser.add_argument("generated_field", type=str, help="Generated output column name in the generated CSV")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file with scores")
    parser.add_argument("tests", type=str, nargs='+', choices=['rouge', 'bleu', 'accuracy', 'length'],
                        help="List of tests to evaluate (rouge, bleu, accuracy, length)")

    # Parse arguments
    args = parser.parse_args()

    # Evaluate the model based on the arguments
    evaluate(args)
