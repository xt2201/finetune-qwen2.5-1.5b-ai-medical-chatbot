import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.qwen_loader import generate_response
from src.utils.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Global variables
model = None
tokenizer = None

def load_config(config_path="configs/config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model():
    global model, tokenizer
    if model is not None:
        return
    
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yml")
    cfg = load_config(config_path)
    
    base_model_name = cfg["model"]["name"]
    # Default to the first experiment's output or a specific one if needed
    # For demo purposes, we'll try to find the best checkpoint or default to base
    adapter_path = os.path.join(os.path.dirname(__file__), "../outputs/final_checkpoint")
    
    # If using experiments, maybe point to a specific one, e.g., lora_r16_alpha32
    # adapter_path = os.path.join(os.path.dirname(__file__), "../outputs/lora_r16_alpha32/final_checkpoint")
    
    logger.info(f"Loading base model: {base_model_name}")
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
        trust_remote_code=True
    )
    
    if os.path.exists(adapter_path):
        logger.info(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        logger.warning(f"Adapter not found at {adapter_path}. Using base model.")
        model = base_model
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def chat(message, history):
    if model is None:
        load_model()
    
    logger.info(f"User: {message}")
    response = generate_response(model, tokenizer, message)
    logger.info(f"Bot: {response[:50]}...")
    
    return response

def main():
    logger.info("Starting Gradio app...")
    
    demo = gr.ChatInterface(
        fn=chat,
        title="Medical Chatbot (Qwen2.5-1.5B Fine-tuned)",
        description="Ask me anything about medical topics.",
        examples=["What are the symptoms of flu?", "How to treat a headache?"],
    )
    
    demo.launch(share=True)

if __name__ == "__main__":
    main()
