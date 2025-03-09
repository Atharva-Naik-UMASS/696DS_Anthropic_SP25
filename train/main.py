from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from unsloth import is_bfloat16_supported
import argparse
import yaml



def ArgParser():

    parser = argparse.ArgumentParser(description="Load arguments from YAML config")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")

    # Add any additional command-line arguments that might override YAML settings
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return argparse.Namespace(**config)

def train():
    args = ArgParser()
    max_seq_length = args.max_seq_length   # Choose any! We auto support RoPE Scaling internally!
    dtype = args.dtype # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = args.load_in_4bit # Use 4bit quantization to reduce memory usage. Can be False.
    data_dir = args.data_dir
    dataset = load_dataset(data_dir, data_files=args.datafile, split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = args.target_modules,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
        bias = args.lora_bias,    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = args.use_gradient_checkpointing, # True or "unsloth" for very long context
        use_rslora = args.use_rslora,  # We support rank stabilized LoRA
        loftq_config = args.loftq_config, # And LoftQ
    )


    if args.dataset_text_field != "false":
        dataset_text_field = args.dataset_text_field
    else:
        dataset_text_field = "text"

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = dataset_text_field,
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False),
        dataset_num_proc = args.dataset_num_proc,
        packing = args.packing, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.num_train_epochs, # Set this for 1 full training run.
            # max_steps = args.max_steps,
            learning_rate = args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = args.logging_steps,
            optim = args.optim,
            weight_decay = args.weight_decay,
            lr_scheduler_type = args.lr_scheduler_type,
            output_dir = args.output_dir,
            report_to = args.report_to # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()
    # To store the merged model
    # model.save_pretrained_merged("/work/pi_wenlongzhao_umass_edu/6/unsloth_test_model", tokenizer, save_method = "merged_16bit")

    # To store the adapters
    try:
        model.save_pretrained(args.output_dir+args.datafile)  
        tokenizer.save_pretrained(args.output_dir+args.datafile)
    except Exception as e:
        print("Cannot find output directory, saving in current directory instead") # Local saving
        model.save_pretrained("adapters/"+args.datafile)
        tokenizer.save_pretrained("adapters/"+args.datafile.split('.')[0])

if __name__ == "__main__":
    train()