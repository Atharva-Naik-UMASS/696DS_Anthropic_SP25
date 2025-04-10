import pandas as pd

df1 = pd.read_csv('/project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/scored_results/scored_results_base_paws_guided.csv')
df2 = pd.read_csv('/project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/scored_results/scored_results_mnli_english_paws_guided.csv')

# Create a boolean mask for rows where the 'score' differs
mask = df1['score'] != df2['score']

comparison = pd.DataFrame({
    'label': df1['label'],
    'Text': df1['Text'],
    'generated_model1': df1['generated'],
    'generated_model2': df2['generated'],
    'score_model1': df1['score'],
    'score_model2': df2['score']
})

# Filter the rows where the scores differ
diff_comparison = comparison[mask]

# Save the filtered differences to an Excel file
diff_comparison.to_excel('differences.xlsx', index=False)

print("The differences have been saved to differences.xlsx")
