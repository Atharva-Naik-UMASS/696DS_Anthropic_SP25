import os
import argparse
import pandas as pd
import time
from google import genai
from google.genai import types

def generate_response(original, api_key, generated):
    print("GENERATE RESPONSE")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    prompt = f"""ONLY OUTPUT 0 OR 1. Compare these:
        Correct Answer: {original}
        AI Response: {generated}
        Are they equivalent? Answer:"""

    for _ in range(3):  # Retry up to 3 times
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    top_p=0.9,
                    max_output_tokens=1,
                    response_mime_type="text/plain",
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"API Error: {str(e)}. Retrying...")
            time.sleep(2)
    return "-1"  # Fallback value

def calculate_similarity_score(response, api_key, generated):
    print("CALC SIMILARITY SCORE")
    gemini_output = generate_response(response, api_key, generated)
    try:
        return int(gemini_output[0])  # Take first character only
    except (ValueError, IndexError):
        return 0  # Treat invalid responses as incorrect

def process_csv(input_csv, api_key, output_csv):
    print("PROCESS CSV")
    df = pd.read_csv(input_csv)
    
    if 'response' not in df.columns or 'generated' not in df.columns:
        raise ValueError("CSV must contain 'response' and 'generated' columns")
    
    # Process with progress
    total = len(df)
    df['score'] = 0
    for idx, row in df.iterrows():
        df.at[idx, 'score'] = calculate_similarity_score(row['response'], api_key, row['generated'])
        if (idx+1) % 10 == 0:
            print(f"Processed {idx+1}/{total} rows...")
    
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    total_score = df['score'].sum()
    print(f"Total Score: {total_score}/{len(df)}")
    print(f"Accuracy: {(total_score/len(df))*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated responses against reference responses")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--api_key", type=str, required=True, help="LLM API key")
    parser.add_argument("--output_csv", type=str, default="scored_results.csv", help="Path to output CSV file")
    args = parser.parse_args()
    print("STARTED EVAL", args.input_csv, args.api_key, args.output_csv)

    process_csv(args.input_csv, args.api_key, args.output_csv)

if __name__ == "__main__":
    print("CALLING MAIN")
    main()