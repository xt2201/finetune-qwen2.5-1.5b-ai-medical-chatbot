import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.qwen_loader import generate_response

def load_finetuned_model(base_model_name, adapter_path):
    """
    Load the base model and the LoRA adapter.
    """
    print(f"Loading base model: {base_model_name}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def compare_models(base_model, finetuned_model, tokenizer, prompts):
    """
    Compare responses from base and fine-tuned models.
    Note: This function assumes you can swap models or have enough VRAM for both.
    If not, run them sequentially in different sessions or reload.
    """
    # For simplicity in this script, we assume we are evaluating the fine-tuned model
    # and maybe the base model if provided.
    pass

def run_qualitative_eval(model, tokenizer, questions):
    """
    Run evaluation on a list of questions.
    """
    print("\n--- Qualitative Evaluation ---")
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"A: {response}")
        print("-" * 50)

if __name__ == "__main__":
    # Example usage
    pass
