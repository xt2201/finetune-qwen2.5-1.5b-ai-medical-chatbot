import logging
import sys
import os
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.theme import Theme

# Custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "header": "bold magenta",
    "panel.border": "blue",
})

console = Console(theme=custom_theme)

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Setup logging with RichHandler (console) and FileHandler.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"project_{timestamp}.log")
    
    # Root logger configuration
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_time=False, show_path=False),
            logging.FileHandler(log_file)
        ]
    )
    
    # Suppress noisy libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    logger = logging.getLogger("setup")
    logger.info(f"Logging initialized. Log file: {log_file}")

def get_logger(name):
    """
    Get a logger instance with the specified name.
    """
    return logging.getLogger(name)

# Keep these for UI elements
def print_header(title):
    """
    Print a styled header panel.
    """
    console.print(Panel(title, style="header", border_style="panel.border"))

def print_success(message):
    """
    Print a success message (using console directly for UI emphasis).
    """
    console.print(f"[success]✔ {message}[/success]")
