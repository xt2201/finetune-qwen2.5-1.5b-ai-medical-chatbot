import sys
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import wandb
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.preprocess import preprocess_and_split
from src.utils.logging_utils import print_header, print_success, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

class LogCallback(TrainerCallback):
    """
    Callback to log metrics to our file logger.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Filter out some verbose keys if needed
            log_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
            logger.info(f"Step {state.global_step}: {log_str}")

def load_config(config_path="configs/config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(experiment_name=None, lora_override=None, resume_from_checkpoint=None):
    print_header(f"LoRA Fine-tuning {f'({experiment_name})' if experiment_name else ''}")
    
    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), "../../configs/config.yml")
    cfg = load_config(config_path)
    
    # Override LoRA config if provided
    if lora_override:
        logger.info(f"Overriding LoRA config: {lora_override}")
        cfg["lora"].update(lora_override)
        
    # Determine output directory
    output_dir = cfg["training"]["output_dir"]
    if experiment_name:
        output_dir = os.path.join(output_dir, experiment_name)
    
    logger.info("Configuration loaded.")
    
    # Apply Environment Overrides (Dataset & Training)
    if os.environ.get("MAX_SAMPLES"):
        logger.info(f"Overriding max_samples: {os.environ.get('MAX_SAMPLES')}")
        cfg["dataset"]["max_samples"] = int(os.environ.get("MAX_SAMPLES"))
        
    if os.environ.get("LEARNING_RATE"):
        logger.info(f"Overriding learning_rate: {os.environ.get('LEARNING_RATE')}")
        cfg["training"]["learning_rate"] = float(os.environ.get("LEARNING_RATE"))
        
    if os.environ.get("NUM_EPOCHS"):
        logger.info(f"Overriding num_train_epochs: {os.environ.get('NUM_EPOCHS')}")
        cfg["training"]["num_train_epochs"] = int(os.environ.get("NUM_EPOCHS"))
    
    # Initialize W&B
    if cfg.get("wandb", {}).get("enabled", True):
        wandb_project = cfg.get("project", {}).get("name", "llm-medical-finetuning")
        # Re-init wandb for each experiment to avoid conflicts
        if wandb.run is not None:
            wandb.finish()
        wandb.init(project=wandb_project, name=experiment_name or "default_run", config=cfg, reinit=True)
    
    # Load Data
    dataset = preprocess_and_split(
        dataset_name=cfg["dataset"]["name"],
        max_samples=cfg["dataset"]["max_samples"],
        test_size=cfg["dataset"]["val_ratio"],
        seed=cfg["dataset"]["seed"]
    )
    
    # Load Tokenizer
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    logger.info(f"Loading model {model_name}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    lora_dropout = cfg["lora"]["lora_dropout"]
    if lora_override and "lora_dropout" in lora_override:
        lora_dropout = lora_override["lora_dropout"]
    
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=lora_dropout,
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"]
    )
    
    # Callbacks
    callbacks = [LogCallback()] # Add our custom logger callback
    if cfg["training"].get("early_stopping", {}).get("enabled", False):
        logger.info("Early stopping enabled.")
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg["training"]["early_stopping"]["patience"],
            early_stopping_threshold=cfg["training"]["early_stopping"]["threshold"]
        ))
    
    # Training Arguments
    warmup_ratio = cfg["training"].get("warmup_ratio", 0.0) or 0.0
    warmup_steps = cfg["training"].get("warmup_steps", 0) or 0
    
    # Only use one of warmup_ratio or warmup_steps (ratio takes precedence)
    if warmup_ratio > 0:
        warmup_steps = 0  # Disable steps when using ratio
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        eval_strategy=cfg["training"]["evaluation_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"] and torch.cuda.is_bf16_supported(),
        optim=cfg["training"]["optim"],
        # dataset_text_field="text", # REMOVED: Allow SFTTrainer to use 'messages' column automatically
        max_length=cfg["model"]["max_length"],
        packing=False,
        load_best_model_at_end=True if cfg["training"].get("early_stopping", {}).get("enabled", False) else False,
        metric_for_best_model="eval_loss" if cfg["training"].get("early_stopping", {}).get("enabled", False) else None,
        neftune_noise_alpha=cfg["training"].get("neftune_noise_alpha"),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "linear"),
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks
    )
    
    # Resume logic
    resume = resume_from_checkpoint if resume_from_checkpoint is not None else cfg["training"].get("resume_from_checkpoint", None)
    
    logger.info(f"Starting training (Resume: {resume})...")
    train_result = trainer.train(resume_from_checkpoint=resume)
    
    logger.info("Saving model...")
    trainer.save_state()
    trainer.save_model(os.path.join(output_dir, "final_checkpoint"))
    
    # Evaluate
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)
    
    print_success(f"Training complete for {experiment_name or 'default'}.")
    wandb.finish()
    
    return metrics

if __name__ == "__main__":
    # Check for env var overrides
    exp_name = os.environ.get("EXPERIMENT_NAME")
    
    lora_override = None
    if os.environ.get("LORA_R"):
        lora_override = {
            "r": int(os.environ.get("LORA_R")),
            "lora_alpha": int(os.environ.get("LORA_ALPHA", 32))
        }
        
    # Apply other overrides directly to config in train function
    # We'll pass them as a separate dict or modify train signature
    # For simplicity, let's modify the train function to accept general overrides
    
    # But first, let's just patch the config loading inside train() by using env vars
    # This is cleaner than changing signature too much
    
    train(experiment_name=exp_name, lora_override=lora_override)
