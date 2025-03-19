from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from unsloth import is_bfloat16_supported
import argparse
import yaml
import os
from datetime import datetime
from peft import PeftConfig


def ArgParser():

    parser = argparse.ArgumentParser(
        description="Load arguments from YAML config")
    parser.add_argument("config_path", type=str,
                        help="Path to YAML config file")

    # Add any additional command-line arguments that might override YAML settings
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    return argparse.Namespace(**config)


def train():
    args = ArgParser()
    # Choose any! We auto support RoPE Scaling internally!
    max_seq_length = args.max_seq_length
    # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    dtype = args.dtype
    # Use 4bit quantization to reduce memory usage. Can be False.
    load_in_4bit = args.load_in_4bit
    data_dir = args.data_dir
    dataset = load_dataset(data_dir, data_files=args.datafile, split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,  # Supports any, but = 0 is optimized
        bias=args.lora_bias,    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        # True or "unsloth" for very long context
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_rslora=args.use_rslora,  # We support rank stabilized LoRA
        loftq_config=args.loftq_config,  # And LoftQ
    )

    if hasattr(args, 'load_adapter') and args.load_adapter:
        adapter_path = args.load_adapter
        if os.path.exists(adapter_path):
            print(f"Loading adapter weights from {adapter_path}")
            model.load_adapter(adapter_path, adapter_name='default')

        else:
            print(
                f"Warning: Adapter path {adapter_path} not found. Training from scratch.")

    if args.dataset_text_field != "false":
        dataset_text_field = args.dataset_text_field
    else:
        dataset_text_field = "text"

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = args.datafile.split(
        '.')[0] if '.' in args.datafile else args.datafile
    run_name = args.wandb_run_name if hasattr(
        args, 'wandb_run_name') else f"{dataset_name}-{current_time}"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False),
        dataset_num_proc=args.dataset_num_proc,
        # Can make training 5x faster for short sequences.
        packing=args.packing,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            # Set this for 1 full training run.
            num_train_epochs=args.num_train_epochs,
            # max_steps = args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            output_dir=os.path.join(args.output_dir, run_name),
            report_to=args.report_to,
            run_name=run_name
        ),
    )

    trainer_stats = trainer.train()
    # To store the merged model
    # model.save_pretrained_merged("/work/pi_wenlongzhao_umass_edu/6/unsloth_test_model", tokenizer, save_method = "merged_16bit")

    # To store the adapters
    try:
        output_path = os.path.join(args.output_dir, run_name)
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Saved model and tokenizer to {output_path}")
    except Exception as e:
        # Local saving
        print("Cannot find output directory, saving in current directory instead")
        path = os.path.join('adapters', run_name)
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)


if __name__ == "__main__":
    train()
