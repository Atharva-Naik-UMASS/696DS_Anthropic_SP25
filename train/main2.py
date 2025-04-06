from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import argparse
import yaml
import os
from datetime import datetime
import torch
from transformers.utils import is_torch_bf16_gpu_available

def ArgParser():
    parser = argparse.ArgumentParser(description="Load arguments from YAML config")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def train():
    args = ArgParser()
    accelerator = Accelerator()
    
    # Load dataset
    data_dir = args.data_dir
    dataset = load_dataset(data_dir, data_files=args.datafile, split="train")
    
    # Load model and tokenizer (using standard transformers instead of Unsloth)
    dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with quantization if needed
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto",  # Let Accelerate handle device mapping
    )
    
    # Prepare model for k-bit training if using quantization
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Load adapter if specified
    if hasattr(args, 'load_adapter') and args.load_adapter:
        adapter_path = args.load_adapter
        if os.path.exists(adapter_path):
            print(f"Loading adapter weights from {adapter_path}")
            model.load_adapter(adapter_path, adapter_name='default')
        else:
            print(f"Warning: Adapter path {adapter_path} not found. Training from scratch.")
    
    # Set up dataset text field
    if args.dataset_text_field != "false":
        dataset_text_field = args.dataset_text_field
    else:
        dataset_text_field = "text"
    
    # Create run name
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = args.datafile.split('.')[0] if '.' in args.datafile else args.datafile
    run_name = args.wandb_run_name if hasattr(args, 'wandb_run_name') else f"{dataset_name}-{current_time}"
    
    # Configure DeepSpeed
    # You'll need a DeepSpeed config file - here's a basic example for ZeRO-3
    deepspeed_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        "bf16": {
            "enabled": is_torch_bf16_gpu_available()
        },
        "fp16": {
            "enabled": not is_torch_bf16_gpu_available()
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size
    }
    
    # Save DeepSpeed config to a file
    import json
    os.makedirs("deepspeed_configs", exist_ok=True)
    with open("deepspeed_configs/zero3.json", "w") as f:
        json.dump(deepspeed_config, f)
    
    # Set up training arguments with DeepSpeed
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not is_torch_bf16_gpu_available(),
        bf16=is_torch_bf16_gpu_available(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        output_dir=os.path.join(args.output_dir, run_name),
        report_to=args.report_to,
        run_name=run_name,
        deepspeed="deepspeed_configs/zero3.json",  # Enable DeepSpeed
        gradient_checkpointing=args.use_gradient_checkpointing,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
        args=training_args,
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model and tokenizer
    try:
        output_path = os.path.join(args.output_dir, run_name)
        os.makedirs(output_path, exist_ok=True)
        # Use accelerator to save on the main process only
        if accelerator.is_main_process:
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"Saved model and tokenizer to {output_path}")
    except Exception as e:
        # Local saving
        if accelerator.is_main_process:
            print("Cannot find output directory, saving in current directory instead")
            path = os.path.join('adapters', run_name)
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

if __name__ == "__main__":
    train()
