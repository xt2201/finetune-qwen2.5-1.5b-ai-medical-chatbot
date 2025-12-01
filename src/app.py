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
        return "Model already loaded."
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yml")
        cfg = load_config(config_path)
        
        base_model_name = cfg["model"]["name"]
        adapter_path = os.path.join(os.path.dirname(__file__), "../outputs/final_checkpoint")
        
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
            
        return "Model loaded successfully!"
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return f"Error loading model: {e}"

def chat(message, history):
    if model is None:
        load_model()
    
    logger.info(f"User: {message}")
    response = generate_response(model, tokenizer, message)
    logger.info(f"Bot: {response[:50]}...")
    
    return response

def main():
    logger.info("Starting Gradio app...")
    
    # Custom Theme
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    
    with gr.Blocks(theme=theme, title="Medical AI Assistant") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    # 🏥 Medical AI Assistant
                    
                    **Powered by Qwen2.5-1.5B (Fine-tuned)**
                    
                    This AI assistant is designed to answer basic medical questions. 
                    
                    > ⚠️ **Disclaimer**: This tool is for educational purposes only and does not constitute professional medical advice. Always consult a doctor for medical concerns.
                    """
                )
                
                with gr.Accordion("Model Information", open=True):
                    gr.Markdown(
                        """
                        - **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
                        - **Fine-tuning**: LoRA (Low-Rank Adaptation)
                        - **Dataset**: Medical Q&A (ruslanmv/ai-medical-chatbot)
                        - **Quantization**: 4-bit (NF4)
                        """
                    )
                
                load_btn = gr.Button("Reload Model", variant="secondary")
                status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                
                load_btn.click(fn=load_model, inputs=[], outputs=[status_text])
                
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat,
                    examples=[
                        "What are the symptoms of flu?",
                        "How to treat a headache?",
                        "What causes high blood pressure?",
                        "Explain diabetes in simple terms."
                    ],
                    title=None,
                    description=None,
                    theme=theme
                )
    
    demo.launch(share=True)

if __name__ == "__main__":
    main()
