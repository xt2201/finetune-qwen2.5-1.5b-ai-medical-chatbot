import json
import os

# Helper to create cells
def new_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def new_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# Title and Intro
cells.append(new_markdown_cell("""# NLP/LLM Engineer Test - Medical Fine-tuning

This notebook demonstrates the complete solution for the NLP/LLM Engineer test.
It covers:
1.  **Setup & Configuration**: Environment and hyperparameters.
2.  **Training**: Fine-tuning Qwen2.5-1.5B-Instruct on a medical dataset using LoRA.
3.  **Evaluation**: Comparing different LoRA configurations (r=8, 16, 32).
4.  **Inference**: Demonstrating the best model on diverse medical queries."""))

# Setup
cells.append(new_markdown_cell("## 1. Setup & Dependencies"))
cells.append(new_code_cell("!pip install -r ../requirements.txt"))
cells.append(new_code_cell("""import sys
import os
import yaml
import torch
from IPython.display import Markdown, display, Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path
sys.path.append(os.path.abspath("../"))
from src.utils.env_utils import setup_env

setup_env()
print("Environment setup complete.")"""))

# Configuration
cells.append(new_markdown_cell("## 2. Configuration\nWe use `configs/config.yml` to manage all hyperparameters."))
cells.append(new_code_cell("""with open("../configs/config.yml", "r") as f:
    config = yaml.safe_load(f)
    print(yaml.dump(config, default_flow_style=False))"""))

# Theory Questions (Task 2)
cells.append(new_markdown_cell("""## Task 2: Theory Questions

### 1. Comparison of Fine-tuning Methods
| Method | Description | Pros | Cons | When to use |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-tuning** | Updates all model parameters. | Best potential performance. | High VRAM & storage cost. Slow. | Small models (<1B) or when you have massive compute resources. |
| **LoRA** | Freezes pre-trained weights, injects trainable rank decomposition matrices. | Low VRAM, fast training, small checkpoint size. | Slightly lower performance ceiling than full FT. | Most standard fine-tuning tasks on consumer GPUs. |
| **QLoRA** | LoRA + 4-bit quantization of base model (NF4). | Lowest VRAM usage. Can run 70B models on single GPU. | Slower training due to dequantization overhead. | Limited VRAM scenarios (e.g., Colab free tier). |

### 2. Important Hyperparameters
- **Learning Rate**: Controls step size during optimization. Too high = unstable; too low = slow convergence. For LoRA, usually higher (2e-4) than full FT.
- **Batch Size**: Number of samples processed before updating weights. Affects gradient stability and VRAM usage.
- **LoRA Rank (r)**: Dimension of the low-rank matrices. Higher r = more trainable params = more capacity, but higher VRAM.
- **LoRA Alpha**: Scaling factor for LoRA weights. Usually set to 2*r. Controls the strength of the adapter.
- **Epochs**: Number of full passes through the dataset. Too many = overfitting; too few = underfitting.
- **Warmup Steps**: Gradually increases LR from 0 to target value to stabilize early training.

### 3. Preventing Overfitting
1.  **Early Stopping**: Stop training when validation loss stops improving.
2.  **Dropout**: Randomly zero out neurons (LoRA dropout) to prevent reliance on specific features.
3.  **Regularization (Weight Decay)**: Penalize large weights to keep the model simple.
"""))

# Training
cells.append(new_markdown_cell("""## 3. Training & Comparison
We run `scripts/train_comparison.py` to train multiple LoRA configurations and compare them.
**Note**: This step may take significant time. If models are already trained, it will skip training and generate reports."""))
cells.append(new_code_cell("!python ../scripts/train_comparison.py"))

# Analysis (Task 4)
cells.append(new_markdown_cell("""## Task 4: Analysis & Conclusion

### 1. Is the model overfitting?
**Answer**: Based on the loss curves (see `outputs/experiment_report.md` or WandB), if the validation loss starts increasing while training loss decreases, the model is overfitting. In our experiments, we used **Early Stopping** and **Dropout (0.2)** to mitigate this. The reduced epoch count (3) also helps.

### 2. Improvements for Performance
- **Data Quality**: Clean and filter the medical dataset further to remove noise.
- **Hyperparameter Tuning**: Run a sweep for Learning Rate and LoRA Rank.
- **Few-shot Prompting**: Include examples in the prompt context during inference.
- **RAG (Retrieval Augmented Generation)**: Connect the model to a medical knowledge base for up-to-date info.

### 3. Production Considerations
- **Latency**: LoRA adapters add minimal latency, but 4-bit quantization might slow down inference slightly.
- **Safety**: Medical advice requires strict guardrails. The model should refuse to answer dangerous questions or include a disclaimer.
- **Monitoring**: Track user feedback and drift in query distribution.
"""))

cells.append(new_markdown_cell("## 4. Results Report\nHere is the summary of our experiments."))
cells.append(new_code_cell("""report_path = "../outputs/experiment_report.md"
if os.path.exists(report_path):
    with open(report_path, "r") as f:
        display(Markdown(f.read()))
else:
    print("Report not found. Please run training first.")"""))

cells.append(new_markdown_cell("### Loss Curves\nVisualizing the training stability."))
cells.append(new_code_cell("""# Display loss curves for the experiments
experiments = ["lora_r16_alpha32_v3", "lora_r16_alpha32_lowlr", "lora_r32_alpha64_v2"]
for exp in experiments:
    img_path = f"../outputs/{exp}/loss_curve.png"
    if os.path.exists(img_path):
        display(Markdown(f"#### {exp}"))
        display(Image(filename=img_path))"""))

# Inference
# Comparative Inference
cells.append(new_markdown_cell("""## 5. Comparative Inference
We compare the generation quality of different LoRA configurations on the same set of questions."""))

cells.append(new_code_cell("""# Load Base Model Once
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading base model: {model_name}...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
print("Base model loaded.")

def generate_response(model, question, max_new_tokens=256):
    prompt = f"<|im_start|>system\\nYou are a helpful medical assistant.<|im_end|>\\n<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Test Questions for Comparison
comparison_questions = [
    "What are the symptoms of diabetes?",
    "How to treat a common cold?"
]

# Iterate and Compare
experiments = ["lora_r16_alpha32_v3", "lora_r16_alpha32_lowlr", "lora_r32_alpha64_v2"]
results = {}

# Initialize PeftModel with the first adapter to get the class wrapper
first_exp = experiments[0]
first_adapter_path = f"../outputs/{first_exp}/final_checkpoint"

if os.path.exists(first_adapter_path):
    print(f"Initializing PeftModel with {first_exp}...")
    model = PeftModel.from_pretrained(base_model, first_adapter_path, adapter_name=first_exp)
else:
    print(f"Warning: {first_adapter_path} not found. Skipping {first_exp}.")
    model = None

if model:
    for exp in experiments:
        adapter_path = f"../outputs/{exp}/final_checkpoint"
        if not os.path.exists(adapter_path):
            continue
            
        if exp != first_exp:
            print(f"Loading adapter: {exp}...")
            model.load_adapter(adapter_path, adapter_name=exp)
            
        print(f"Switching to adapter: {exp}")
        model.set_adapter(exp)
        
        print(f"Generating responses for {exp}...")
        exp_results = []
        for q in comparison_questions:
            ans = generate_response(model, q)
            exp_results.append(ans)
        results[exp] = exp_results

    # Print Comparison
    for i, q in enumerate(comparison_questions):
        print(f"\\n=== Question: {q} ===")
        for exp, answers in results.items():
            print(f"\\n[{exp}]:\\n{answers[i]}")
            print("-" * 30)
"""))

cells.append(new_markdown_cell("""## 6. Inference with Best Model
We load the best performing checkpoint (dynamically detected) and test it on diverse medical queries."""))

cells.append(new_code_cell("""# Dynamic Best Model Detection
import json
import glob

def get_best_model_path(outputs_dir="../outputs"):
    best_loss = float("inf")
    best_exp = None
    
    # Iterate over all experiment directories
    exp_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    for exp in exp_dirs:
        trainer_state_path = os.path.join(outputs_dir, exp, "final_checkpoint", "trainer_state.json")
        if not os.path.exists(trainer_state_path):
             # Fallback to checking the directory itself if final_checkpoint doesn't have it (rare)
             trainer_state_path = os.path.join(outputs_dir, exp, "trainer_state.json")
        
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, "r") as f:
                    data = json.load(f)
                    # Check for best_metric (usually eval_loss)
                    loss = data.get("best_metric")
                    if loss is None and "log_history" in data:
                        # Try to find min eval_loss in history if best_metric not set
                        eval_losses = [x["eval_loss"] for x in data["log_history"] if "eval_loss" in x]
                        if eval_losses:
                            loss = min(eval_losses)
                            
                    if loss is not None and loss < best_loss:
                        best_loss = loss
                        best_exp = exp
            except Exception as e:
                print(f"Error reading {trainer_state_path}: {e}")
                
    if best_exp:
        return os.path.join(outputs_dir, best_exp, "final_checkpoint"), best_exp
    return None, None

adapter_path, best_exp_name = get_best_model_path()

if adapter_path:
    print(f"Automatically selected best model: {best_exp_name}")
    print(f"Adapter path: {adapter_path}")
    
    # Load the best adapter
    # Note: 'model' is already a PeftModel from previous cell. We can just load this new one.
    if 'model' in locals() and model is not None:
        print(f"Loading best adapter {best_exp_name} into existing model...")
        try:
            model.load_adapter(adapter_path, adapter_name="best_model")
            model.set_adapter("best_model")
        except Exception as e:
             print(f"Adapter might already exist or error: {e}. Trying to set it if exists.")
             if best_exp_name in model.peft_config:
                 model.set_adapter(best_exp_name)
             else:
                 # Fallback: Re-init if something is wrong
                 print("Re-initializing model for best checkpoint...")
                 model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name="best_model")
    else:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
    model.eval()
    print("Best model loaded successfully!")

else:
    print("Could not find any trained models.")
"""))

cells.append(new_code_cell("""def generate_response(question, max_new_tokens=256):
    prompt = f"<|im_start|>system\\nYou are a helpful medical assistant.<|im_end|>\\n<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Diverse Use Cases
test_cases = [
    "What are the early warning signs of diabetes?",
    "I have a severe headache and sensitivity to light. What could it be?",
    "Explain the difference between viral and bacterial infections.",
    "What is the recommended treatment for a sprained ankle?",
    "Can you explain how vaccines work to a 5-year-old?"
]

print("--- Inference Results ---")
for i, case in enumerate(test_cases, 1):
    print(f"\\nCase {i}: {case}")
    response = generate_response(case)
    print(f"Response: {response}")
    print("-" * 50)"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/notebook.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
