import sys
import os
from datasets import load_dataset
import pandas as pd
import numpy as np
from rich.table import Table

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logging_utils import console, print_header, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def load_medical_dataset(dataset_name="ruslanmv/ai-medical-chatbot", max_samples=None):
    """
    Load the medical dataset from Hugging Face.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    if max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        dataset = dataset.select(range(min(len(dataset), max_samples)))
        
    return dataset

def explore_dataset(dataset):
    """
    Print statistics about the dataset using Rich tables.
    """
    print_header("Dataset Exploration")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Features: {dataset.features}")
    
    # Convert to pandas for easier analysis
    df = dataset.to_pandas()
    
    question_col = 'Description' if 'Description' in df.columns else df.columns[0]
    answer_col = 'Doctor' if 'Doctor' in df.columns else df.columns[1]
    
    logger.info(f"Using '{question_col}' as Question and '{answer_col}' as Answer")
    
    df['q_len'] = df[question_col].astype(str).apply(len)
    df['a_len'] = df[answer_col].astype(str).apply(len)
    
    # Create Stats Table
    table = Table(title="Length Statistics (Characters)")
    table.add_column("Metric", style="cyan")
    table.add_column("Question", style="magenta")
    table.add_column("Answer", style="green")
    
    for metric in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        q_val = df['q_len'].describe()[metric]
        a_val = df['a_len'].describe()[metric]
        table.add_row(metric, f"{q_val:.2f}", f"{a_val:.2f}")
        
    console.print(table)
    
    # Sample Entry
    print_header("Sample Entry")
    console.print(f"[bold magenta]Q:[/bold magenta] {df.iloc[0][question_col]}")
    console.print(f"[bold green]A:[/bold green] {df.iloc[0][answer_col]}")
    
    return df

if __name__ == "__main__":
    ds = load_medical_dataset()
    explore_dataset(ds)
