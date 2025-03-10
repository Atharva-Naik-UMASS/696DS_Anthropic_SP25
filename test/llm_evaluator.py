import os
import argparse
import pandas as pd
import time
from openai import OpenAI
import asyncio
import aiohttp
import json

async def generate_response(session, original, api_key, generated):
    print("GENERATE RESPONSE")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""ONLY OUTPUT 0 OR 1. Compare these:
        Correct Answer: {original}
        AI Response: {generated}
        Are they equivalent? Answer:"""

    data = json.dumps({
        "model": "deepseek/deepseek-r1-zero:free",
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ]
    })

    for _ in range(3):  # Retry up to 3 times
        try:
            async with session.post(url, data=data, headers=headers) as response:
                response_data = await response.json()
                print(response_data)
                result = response_data['choices'][0]['message']['content'].strip()
                print(result)
                return result
        except Exception as e:
            print(f"API Error: {str(e)}. Retrying...")
            time.sleep(2)
    return "-1"  # Fallback value

async def calculate_similarity_score(session, response, api_key, generated):
    print("CALC SIMILARITY SCORE")
    gemini_output = await generate_response(session, response, api_key, generated)
    try:
        return int(gemini_output[0])  # Take first character only
    except (ValueError, IndexError):
        return 0  # Treat invalid responses as incorrect

async def process_csv(input_csv, api_key, output_csv):
    print("PROCESS CSV")
    df = pd.read_csv(input_csv)
    
    if 'response' not in df.columns or 'generated' not in df.columns:
        raise ValueError("CSV must contain 'response' and 'generated' columns")
    
    # Process with progress
    total = len(df)
    df['score'] = 0
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in df.iterrows():
            task = asyncio.ensure_future(calculate_similarity_score(session, row['response'], api_key, row['generated']))
            tasks.append((idx, task))
        for idx, task in tasks:
            df.at[idx, 'score'] = await task
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
    print("STARTED EVAL")

    asyncio.run(process_csv(args.input_csv, args.api_key, args.output_csv))
    # process_csv(args.input_csv, args.api_key, args.output_csv)

if __name__ == "__main__":
    print("CALLING MAIN")
    main()