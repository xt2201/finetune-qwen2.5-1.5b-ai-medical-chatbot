# Theory: Fine-tuning LLMs

## 1. Comparison: Full Fine-tuning vs LoRA vs QLoRA

| Feature | Full Fine-tuning | LoRA (Low-Rank Adaptation) | QLoRA (Quantized LoRA) |
| :--- | :--- | :--- | :--- |
| **Mechanism** | Updates **all** model parameters. | Freezes pre-trained weights. Injects trainable rank decomposition matrices into layers. | Same as LoRA, but freezes base model in **4-bit** precision and uses paged optimizers. |
| **VRAM Usage** | Very High (requires storing gradients/optimizer states for all params). | Medium (significantly reduced optimizer states). | Very Low (base model takes ~1/4 memory compared to FP16). |
| **Training Speed** | Slowest (more params to update). | Fast. | Slightly slower than LoRA due to dequantization overhead, but enables training on smaller GPUs. |
| **Storage** | Huge (saves full model copy). | Tiny (saves only adapter weights, ~MBs). | Tiny (saves only adapter weights). |
| **Use Case** | Huge resources available; need to change model behavior drastically. | Limited VRAM; domain adaptation; need to switch adapters easily. | Very limited VRAM (e.g., consumer GPUs); same benefits as LoRA. |

**Summary:**
- Use **Full Fine-tuning** only when you have massive compute clusters and need to fundamentally change the model's knowledge base.
- Use **LoRA** for most standard fine-tuning tasks on enterprise GPUs (A100, A10).
- Use **QLoRA** for fine-tuning large models (7B, 70B) on consumer hardware (T4, L4, RTX 3090/4090) or Colab.

## 2. Hyperparameters Explanation

- **`learning_rate` (LR)**: Controls the step size during gradient descent.
  - *Too high*: Model diverges or oscillates.
  - *Too low*: Training is too slow or gets stuck in local minima.
  - *LoRA/QLoRA*: Usually higher than full FT (e.g., 2e-4 vs 1e-5).

- **`batch_size`**: Number of samples processed before updating weights.
  - *Impact*: Larger batch size = more stable gradients but higher VRAM.
  - *Intuition*: Like how many opinions you listen to before changing your mind.

- **`lora_r` (Rank)**: The dimension of the low-rank matrices.
  - *Impact*: Higher `r` = more trainable parameters = more capacity to learn, but higher VRAM and risk of overfitting.
  - *Typical values*: 8, 16, 32, 64.

- **`lora_alpha`**: Scaling factor for LoRA weights.
  - *Formula*: Update = $W + \frac{\alpha}{r} \Delta W$.
  - *Rule of thumb*: Often set $\alpha = 2 \times r$ or $\alpha = r$. It controls how much "influence" the adapter has over the base model.

- **`num_epochs`**: Number of times the model sees the entire dataset.
  - *Impact*: Too few = underfitting; Too many = overfitting.
  - *Typical*: 1-3 epochs for SFT (Supervised Fine-Tuning).

- **`warmup_steps`**: Number of steps to linearly increase LR from 0 to target LR at the start.
  - *Purpose*: Stabilizes training early on when gradients might be erratic.

## 3. Overfitting & Prevention Strategies

**Overfitting** occurs when the model memorizes the training data instead of learning general patterns. It performs well on training data but poorly on new/validation data.

**Prevention Strategies:**

1.  **Early Stopping**: Monitor validation loss. Stop training when validation loss stops decreasing or starts increasing for a set number of steps (patience).
    - *Example*: Stop if val_loss doesn't improve for 3 evaluations.

2.  **Regularization (Weight Decay / Dropout)**:
    - *Weight Decay*: Adds a penalty to the loss function for large weights, encouraging simpler models.
    - *Dropout*: Randomly "turns off" neurons during training to prevent reliance on specific features. `lora_dropout` is commonly used.

3.  **Data Augmentation / More Data**: Increasing the diversity of training data makes it harder for the model to memorize exact examples.
    - *Example*: Paraphrasing questions or adding noise to inputs.

4.  **Smaller Learning Rate / Rank**: Reducing the capacity of the model (lower `r`) or making smaller updates (lower LR) can prevent it from fitting noise.
