import json
import matplotlib.pyplot as plt
import sys
import os

def plot_loss(trainer_state_path, output_path):
    with open(trainer_state_path, 'r') as f:
        data = json.load(f)
    
    log_history = data['log_history']
    
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Training Loss')
    plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (NEFTune + Cosine)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_loss.py <trainer_state_path> <output_path>")
        sys.exit(1)
    plot_loss(sys.argv[1], sys.argv[2])
