import sys
import os
import yaml
import pandas as pd
import subprocess
from rich.progress import track
from dotenv import load_dotenv
import torch
import gc
import math
import matplotlib.pyplot as plt
import json

load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logging_utils import setup_logging, get_logger, print_header

setup_logging()
logger = get_logger(__name__)

def load_config(config_path="configs/config.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_report(results, output_file="outputs/experiment_report.md"):
    """
    Generate a Markdown report from the experiment results.
    """
    df = pd.DataFrame(results)
    
    markdown = "# Experiment Report\n\n"
    markdown += "## Summary Table\n\n"
    markdown += df.to_markdown(index=False)
    markdown += "\n\n"
    
    markdown += "## Detailed Metrics\n\n"
    for res in results:
        markdown += f"### {res['Experiment']}\n"
        for k, v in res.items():
            if k != "Experiment" and k != "Loss Curve":
                markdown += f"- **{k}**: {v}\n"
        
        if "Loss Curve" in res:
            markdown += f"\n![Loss Curve]({res['Loss Curve']})\n"
        markdown += "\n"
        
    with open(output_file, "w") as f:
        f.write(markdown)
        
    logger.info(f"Report generated at {output_file}")
    logger.info("\n" + df.drop(columns=["Loss Curve"], errors="ignore").to_string(index=False))

def plot_loss_curve(log_history, output_path):
    """
    Plot training and evaluation loss from log history.
    """
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    
    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
            
    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_loss, label="Training Loss")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="Validation Loss")
        
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def get_metrics_and_history(output_dir):
    """
    Read metrics and log history from trainer_state.json.
    """
    # Try final_checkpoint first
    state_path = os.path.join(output_dir, "final_checkpoint", "trainer_state.json")
    
    if not os.path.exists(state_path):
        # Find latest checkpoint
        try:
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Sort by number
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = checkpoints[-1]
                state_path = os.path.join(output_dir, latest_checkpoint, "trainer_state.json")
        except Exception:
            pass
    
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
            log_history = state.get("log_history", [])
            
            metrics = {}
            if log_history:
                # Iterate backwards to find the last entry with eval_loss and train loss
                for log in reversed(log_history):
                    if "eval_loss" in log and "eval_loss" not in metrics:
                        metrics.update(log)
                    if "loss" in log and "train_loss" not in metrics:
                        metrics["train_loss"] = log["loss"]
                    if "epoch" in log and "epoch" not in metrics:
                        metrics["epoch"] = log["epoch"]
                        
                    if "eval_loss" in metrics and "train_loss" in metrics:
                        break
            
            return metrics, log_history
            
    return {}, []

def run_experiments():
    print_header("Running LoRA Experiments (Subprocess)")
    
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yml")
    cfg = load_config(config_path)
    
    experiments = cfg.get("experiments", [])
    if not experiments:
        logger.warning("No experiments found in config.yml")
        return

    results = []
    
    for exp in experiments:
        exp_name = exp["name"]
        lora_config = exp["lora"]
        
        # Resolve output_dir relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir_base = cfg["training"]["output_dir"]
        if output_dir_base.startswith("./"):
            output_dir_base = output_dir_base[2:]
        
        output_dir = os.path.join(project_root, output_dir_base, exp_name)
        final_checkpoint = os.path.join(output_dir, "final_checkpoint")
        
        logger.info(f"Checking for results in: {output_dir}")
        
        if os.path.exists(final_checkpoint):
            logger.info(f"Experiment {exp_name} already completed. Skipping training.")
            metrics, log_history = get_metrics_and_history(output_dir)
            
            # Calculate Perplexity
            ppl = "N/A"
            if metrics.get("eval_loss") != "N/A" and metrics.get("eval_loss") is not None:
                try:
                    ppl = math.exp(metrics["eval_loss"])
                except OverflowError:
                    ppl = float("inf")
            
            # Generate Plot
            plot_path = os.path.join(output_dir, "loss_curve.png")
            if log_history:
                plot_loss_curve(log_history, plot_path)
            
            result = {
                "Experiment": exp_name,
                "Train Loss": metrics.get("train_loss", "N/A"),
                "Eval Loss": metrics.get("eval_loss", "N/A"),
                "Perplexity": ppl,
                "Epochs": metrics.get("epoch", "N/A"),
                "Loss Curve": os.path.relpath(plot_path, os.path.dirname("outputs/experiment_report.md"))
            }
            results.append(result)
            continue
            
        # Prepare env vars
        env = os.environ.copy()
        env["EXPERIMENT_NAME"] = exp_name
        env["LORA_R"] = str(lora_config["r"])
        env["LORA_ALPHA"] = str(lora_config["lora_alpha"])
        
        # Add overrides for dataset and training
        if "dataset" in exp and "max_samples" in exp["dataset"]:
            env["MAX_SAMPLES"] = str(exp["dataset"]["max_samples"])
            
        if "training" in exp:
            if "learning_rate" in exp["training"]:
                env["LEARNING_RATE"] = str(exp["training"]["learning_rate"])
            if "num_train_epochs" in exp["training"]:
                env["NUM_EPOCHS"] = str(exp["training"]["num_train_epochs"])
        
        cmd = [sys.executable, "src/models/train_lora.py"]
        
        try:
            subprocess.run(cmd, env=env, check=True, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            
            # Collect metrics from output file
            output_dir = os.path.join(cfg["training"]["output_dir"], exp_name) # Re-resolve for safety
            # Resolve output_dir relative to project root again just in case
            output_dir = os.path.join(project_root, output_dir_base, exp_name)
            
            metrics, log_history = get_metrics_and_history(output_dir)
            
            # Calculate Perplexity
            ppl = "N/A"
            if metrics.get("eval_loss") != "N/A" and metrics.get("eval_loss") is not None:
                try:
                    ppl = math.exp(metrics["eval_loss"])
                except OverflowError:
                    ppl = float("inf")
            
            # Generate Plot
            plot_path = os.path.join(output_dir, "loss_curve.png")
            if log_history:
                plot_loss_curve(log_history, plot_path)
            
            result = {
                "Experiment": exp_name,
                "Train Loss": metrics.get("train_loss", "N/A"),
                "Eval Loss": metrics.get("eval_loss", "N/A"),
                "Perplexity": ppl,
                "Epochs": metrics.get("epoch", "N/A"),
                "Loss Curve": os.path.relpath(plot_path, os.path.dirname("outputs/experiment_report.md"))
            }
            results.append(result)
            
            logger.info(f"Experiment {exp_name} completed successfully.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment {exp_name} failed with exit code {e.returncode}")
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            
    if results:
        generate_report(results)

if __name__ == "__main__":
    run_experiments()
