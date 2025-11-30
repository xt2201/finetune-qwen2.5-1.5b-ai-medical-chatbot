import os
from dotenv import load_dotenv

def setup_env():
    """
    Load environment variables from .env file.
    """
    load_dotenv()
    
    # Set HF_HOME if provided
    if os.getenv("HF_HOME"):
        print(f"Setting HF_HOME to {os.getenv('HF_HOME')}")
        
    # Check for W&B key
    if os.getenv("WANDB_API_KEY"):
        print("W&B API Key found.")
    else:
        print("W&B API Key not found in .env (optional).")
