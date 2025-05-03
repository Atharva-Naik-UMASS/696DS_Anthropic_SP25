import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import argparse
import yaml
from datetime import datetime
from functools import partial

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import default_data_collator


def parse_args():
    parser = argparse.ArgumentParser(description="Load arguments from YAML config")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def preprocess_for_label_only(examples, tokenizer, text_column, max_seq_length):
    inputs = examples[text_column]
    
    # Process inputs first to get tokenized outputs
    tokenized = tokenizer(
        inputs, 
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None
    )
    
    # Initialize labels with -100 (masked tokens that don't contribute to loss)
    labels_list = []
    for idx, text in enumerate(inputs):
        lines = text.strip().split("\n")
        label_line = lines[-1].replace("<|end_of_text|>", "").strip()
        label_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label_line))
        input_ids = tokenized["input_ids"][idx]

        # initialize all masked
        labels = [-100] * len(input_ids)
        # locate label tokens
        for i in range(len(input_ids) - len(label_ids) + 1):
            if input_ids[i : i + len(label_ids)] == label_ids:
                for offset in range(len(label_ids)):
                    labels[i + offset] = input_ids[i + offset]
                break
        labels_list.append(labels)

    tokenized["labels"] = labels_list
    # Ensure labels length matches input_ids
    assert len(tokenized["labels"][0]) == len(tokenized["input_ids"][0]), "Labels and input_ids length mismatch"
    return tokenized


def train():
    args = parse_args()
    dataset = load_dataset(args.data_dir, data_files=args.datafile, split="train")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    # Optionally load existing adapter
    if getattr(args, 'load_adapter', None):
        if os.path.isdir(args.load_adapter):
            model.load_adapter(args.load_adapter, adapter_name='default')
        else:
            print(f"Adapter path {args.load_adapter} not found; training from scratch.")

    # Prepare dataset
    preprocess_fn = partial(preprocess_for_label_only, tokenizer=tokenizer, text_column=args.dataset_text_field, max_seq_length=args.max_seq_length)
    tokenized_ds = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=args.dataset_num_proc,
    )

    # Setup trainer
    run_name = args.wandb_run_name if hasattr(args, 'wandb_run_name') else f"finetune-{datetime.now():%Y%m%d_%H%M%S}"
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, run_name),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=run_name,
        max_seq_length=args.max_seq_length,
        remove_unused_columns=False,
        dataset_text_field=args.dataset_text_field,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,  # Use the tokenized_ds with our custom preprocessing
        data_collator=default_data_collator,  # Use the default collator since we already created labels
        args=training_args,
    )

    trainer.train()

    # Save adapter
    save_path = os.path.join(args.output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved adapter and tokenizer at {save_path}")


if __name__ == "__main__":
    train()
