#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=20GB
#SBATCH --time=04:00:00

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/6/envs/anthro_finetune
# python eval_models.py --model_dir /datasets/ai/llama3/meta-llama/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
#                  --adapter_dir /work/pi_wenlongzhao_umass_edu/6/adapters/financial_sentiment_test \
#                  --test_csv /work/pi_wenlongzhao_umass_edu/6/datasets/financial-sentiment-analysis/test_data.csv \
#                  --target_field Sentiment \
#                  --input_fields Formatted \
#                  --output_csv generated_output_adapter.csv

python test/eval_models.py /project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/generated/generated_output_paws_adapter_mnli_english_guided.csv \
                /project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/generated/generated_output_paws_adapter_mnli_english_guided.csv \
                label \
                generated \
                output_accuracy.csv \
                /project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/scored_results/scored_results_paws_mnli_english_guided.csv \
                accuracy
                 