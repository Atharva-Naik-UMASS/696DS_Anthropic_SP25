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


def preprocess_mask_labels(examples, tokenizer, text_column, max_seq_length):
    # First find label spans in original text
    label_spans = []
    processed_texts = []
    for text in examples[text_column]:
        # Find label position between last \n and <|end_of_text|>
        end_pos = text.find("<|end_of_text|>")
        if end_pos == -1:
            # No label found
            label_spans.append((0, 0))
            processed_texts.append(text)
            continue
            
        # Find last newline before end token
        last_newline = text.rfind("\n", 0, end_pos)
        if last_newline == -1:
            # No newline - label starts at beginning
            label_start = 0
        else:
            label_start = last_newline + 1  # Skip the \n
            
        label_end = end_pos
        label_spans.append((label_start, label_end))
        processed_texts.append(text)  # Keep original text

    # Tokenize with offset mapping
    tokenized = tokenizer(
        processed_texts,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    
    # Create labels mask
    labels = []
    for i in range(len(tokenized["input_ids"])):
        input_ids = tokenized["input_ids"][i]
        offsets = tokenized["offset_mapping"][i]
        label_start, label_end = label_spans[i]
        
        example_labels = []
        for token_idx, (char_start, char_end) in enumerate(offsets):
            # Mask if token overlaps with label span
            if char_start >= label_start and char_end <= label_end:
                example_labels.append(-100)
            else:
                example_labels.append(input_ids[token_idx])
                
        # Also mask padding tokens
        attention_mask = tokenized["attention_mask"][i]
        for j in range(len(example_labels)):
            if attention_mask[j] == 0:
                example_labels[j] = -100
                
        labels.append(example_labels)
    
    tokenized["labels"] = labels
    del tokenized["offset_mapping"]  # Remove unused data
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

    # Prepare dataset using the new preprocessing function
    preprocess_fn = partial(preprocess_mask_labels, tokenizer=tokenizer, text_column=args.dataset_text_field, max_seq_length=args.max_seq_length)
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