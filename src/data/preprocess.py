import sys
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split
from rich.progress import track

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.load_dataset import load_medical_dataset
from src.utils.logging_utils import print_header, print_success, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def format_instruction(sample):
    """
    Format a sample into ChatML style.
    """
    # Adjust column names based on dataset
    question = sample.get('Description', sample.get('question', ''))
    answer = sample.get('Doctor', sample.get('answer', ''))
    
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": str(question)},
        {"role": "assistant", "content": str(answer)}
    ]
    
    return {"messages": messages}

def preprocess_and_split(dataset_name="ruslanmv/ai-medical-chatbot", max_samples=1000, test_size=0.1, seed=42):
    """
    Load, format, and split the dataset.
    """
    print_header("Data Preprocessing")
    
    dataset = load_medical_dataset(dataset_name, max_samples)
    
    # Format
    logger.info("Formatting dataset...")
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Split
    logger.info(f"Splitting dataset (test_size={test_size})...")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    logger.info(f"Train size: {len(split_dataset['train'])}")
    logger.info(f"Test/Val size: {len(split_dataset['test'])}")
    
    # Calculate Statistics
    logger.info("Calculating dataset statistics...")
    lengths = []
    for sample in split_dataset['train']:
        # Estimate length based on characters or words
        # Here we use character length of the conversation
        content = ""
        for msg in sample['messages']:
            content += msg['content']
        lengths.append(len(content))
        
    if lengths:
        import numpy as np
        avg_len = np.mean(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        median_len = np.median(lengths)
        
        logger.info(f"Average Length (chars): {avg_len:.2f}")
        logger.info(f"Length Distribution - Min: {min_len}, Max: {max_len}, Median: {median_len}")
    
    return split_dataset

if __name__ == "__main__":
    ds = preprocess_and_split()
    print_success("Preprocessing complete.")
