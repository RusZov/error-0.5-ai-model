#!/usr/bin/env python3
"""
Main entry point for Error 0.5 AI Model Experiment.

Usage:
    python main.py --prompts "What is 2+2?" "Is the sky blue?"
    python main.py --config config.yaml --n-repetitions 50
    python main.py --compare --temperature 0.5 --temp-b 1.5
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.experiment import main

if __name__ == "__main__":
    main()
