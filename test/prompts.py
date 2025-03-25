prompts = {
    "prompt_sudharshan": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Your job is to check whether the AI's answer is correct. Compare it with the correct answer and score it as either 0 if the AI's answer is wrong or 1 if it is correct. DO NOT provide any explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>
### Original Question:
{row['Text'].replace('<|begin_of_text|>', '').replace('Answer:', '')}
### Correct Answer: 
{row[groundtruth_column]}
### AI's Answer:
{row['generated']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Score: """
}