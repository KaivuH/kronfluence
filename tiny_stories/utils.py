
import argparse
import logging
from datetime import timedelta
import os
import sys

# Add the parent directory of 'kronfluencer' to the Python path
sys.path.append('/data/scratch/kaivuh')

# Now import using an absolute import
from kronfluencer.kron.analyzer import Analyzer

use_local = True  # Set this to False to use the original kronfluence

# def import_local_kronfluence():
#     local_path = '/data/scratch/kaivuh/kronfluencer/kronfluence'
#     if local_path not in sys.path:
#         sys.path.insert(0, local_path)
    
#     # Remove any existing kronfluence from sys.modules
#     if 'kronfluence' in sys.modules:
#         del sys.modules['kronfluence']
    
#     try:
#         kf = importlib.import_module('kronfluence')
#         print(f"Using local kronfluence from {local_path}")
#         return kf
#     except ImportError as e:
#         print(f"Failed to import local kronfluence from {local_path}")
#         print(f"Error: {e}")
#         return None

# def import_installed_kronfluence():
#     try:
#         import kronfluence as kf
#         print("Using installed kronfluence")
#         return kf
#     except ImportError as e:
#         print("Failed to import installed kronfluence")
#         print(f"Error: {e}")
#         return None

if __name__ == '__main__':
    # Choose which version to import
    analyzer = Analyzer()
    print(analyzer)