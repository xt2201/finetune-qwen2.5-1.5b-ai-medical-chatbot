import matplotlib.pyplot as plt
import json
import os
import math

def plot_loss_curve(log_history, output_path="loss_curve.png"):
    """
    Plot training and validation loss from trainer log history.
    """
    train_loss = []
    val_loss = []
    steps = []
    
    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_loss.append((entry["step"], entry["loss"]))
        if "eval_loss" in entry and "step" in entry:
            val_loss.append((entry["step"], entry["eval_loss"]))
            
    plt.figure(figsize=(10, 6))
    
    if train_loss:
        steps_t, loss_t = zip(*train_loss)
        plt.plot(steps_t, loss_t, label="Training Loss")
        
    if val_loss:
        steps_v, loss_v = zip(*val_loss)
        plt.plot(steps_v, loss_v, label="Validation Loss")
        
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")

def calculate_perplexity(loss):
    """
    Calculate perplexity from loss.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

if __name__ == "__main__":
    pass
