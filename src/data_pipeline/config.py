# Legacy config stub - all configuration moved to project root config.py
# Import Config from root for any code that still references this module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
