import sys
import os
import yaml
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.qwen_loader import generate_response
from src.evaluation.metrics import plot_loss_curve
from src.utils.logging_utils import console, print_header, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def load_config(config_path="configs/config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_base_model(model_name):
    logger.info(f"Loading base model: {model_name}")
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def load_adapter(base_model, adapter_path):
    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model

def calculate_perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yml")
    cfg = load_config(config_path)
    
    model_name = cfg["model"]["name"]
    # Default to final_checkpoint, but could be parameterized
    adapter_path = os.path.join(cfg["training"]["output_dir"], "final_checkpoint")
    
    # 1. Load Base Model
    base_model, tokenizer = load_base_model(model_name)
    
    questions = cfg["evaluation"]["test_questions"]
    
    print_header("Evaluation: Base vs Fine-tuned")
    
    # Store base responses
    base_responses = []
    logger.info("Generating responses with Base Model...")
    for q in questions:
        base_responses.append(generate_response(base_model, tokenizer, q))
        
    # 2. Load Fine-tuned Model
    if os.path.exists(adapter_path):
        ft_model = load_adapter(base_model, adapter_path)
        
        ft_responses = []
        logger.info("Generating responses with Fine-tuned Model...")
        for q in questions:
            ft_responses.append(generate_response(ft_model, tokenizer, q))
            
        # 3. Compare and Display
        for i, q in enumerate(questions):
            console.print(Panel(f"[bold]Q{i+1}:[/bold] {q}", style="blue"))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Model", style="dim", width=12)
            table.add_column("Response")
            
            table.add_row("Base", Markdown(base_responses[i]))
            table.add_row("Fine-tuned", Markdown(ft_responses[i]))
            
            console.print(table)
            console.print("")
            
        # 4. Quantitative Metrics
        trainer_state_path = os.path.join(adapter_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            import json
            with open(trainer_state_path, "r") as f:
                state = json.load(f)
                
                # Plot Loss
                plot_loss_curve(state["log_history"], os.path.join(cfg["training"]["output_dir"], "loss_curve.png"))
                logger.info("Loss curve saved.")
                
                # Calculate Perplexity (using last eval loss)
                final_eval_loss = None
                for log in reversed(state["log_history"]):
                    if "eval_loss" in log:
                        final_eval_loss = log["eval_loss"]
                        break
                
                if final_eval_loss:
                    ppl = calculate_perplexity(final_eval_loss)
                    console.print(Panel(f"Final Validation Loss: {final_eval_loss:.4f}\nPerplexity: {ppl:.4f}", title="Quantitative Metrics", style="red"))
                else:
                    logger.warning("No eval_loss found in logs.")
        else:
            logger.warning("trainer_state.json not found.")
            
    else:
        logger.warning(f"Adapter not found at {adapter_path}. Skipping FT evaluation.")

if __name__ == "__main__":
    main()
