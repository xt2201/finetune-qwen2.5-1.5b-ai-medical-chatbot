import sys
import os
import time
import torch
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.qwen_loader import load_model_and_tokenizer, generate_response
from src.utils.logging_utils import console, print_header, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def main():
    prompts = [
        "What is machine learning?",
        "Explain neural networks in simple terms.",
        "What are the benefits of fine-tuning LLMs?"
    ]
    
    print_header("Qwen2.5-1.5B-Instruct Inference")
    
    # Measure VRAM before loading
    vram_before = 0
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**2
    
    start_load = time.time()
    model, tokenizer = load_model_and_tokenizer()
    load_time = time.time() - start_load
    
    vram_after = 0
    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / 1024**2
        
    # Stats Table
    stats_table = Table(title="Model Loading Stats")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Load Time", f"{load_time:.2f} s")
    stats_table.add_row("VRAM (Before)", f"{vram_before:.2f} MB")
    stats_table.add_row("VRAM (After)", f"{vram_after:.2f} MB")
    stats_table.add_row("VRAM Usage", f"{vram_after - vram_before:.2f} MB")
    console.print(stats_table)
    
    logger.info(f"Model loaded in {load_time:.2f}s. VRAM usage: {vram_after - vram_before:.2f} MB")
    
    for i, prompt in enumerate(prompts, 1):
        console.print(Panel(f"[bold]Prompt {i}:[/bold] {prompt}", style="blue"))
        logger.info(f"Processing Prompt {i}: {prompt}")
        
        start_gen = time.time()
        response = generate_response(model, tokenizer, prompt)
        end_gen = time.time()
        
        console.print(Panel(Markdown(response), title="Response", style="green"))
        console.print(f"[dim]Time taken: {end_gen - start_gen:.2f} seconds[/dim]\n")
        logger.info(f"Response generated in {end_gen - start_gen:.2f}s")

if __name__ == "__main__":
    main()
