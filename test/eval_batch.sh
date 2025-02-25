#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=20GB
#SBATCH --time=04:00:00

module load conda/latest
conda activate ./../../../envs/anthro_finetune
# python eval_models.py --model_dir /datasets/ai/llama3/meta-llama/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
#                  --adapter_dir /work/pi_wenlongzhao_umass_edu/6/adapters/financial_sentiment_test \
#                  --test_csv /work/pi_wenlongzhao_umass_edu/6/datasets/financial-sentiment-analysis/test_data.csv \
#                  --target_field Sentiment \
#                  --input_fields Formatted \
#                  --output_csv generated_output_adapter.csv

python eval_models.py ./../../../utils/generated_csv/generated_output_adapter.csv \
                ./../../../utils/generated_csv/generated_output_unsloth.csv \
                Sentiment \
                Sentiment \
                output_accuracy.csv \
                accuracy
                 