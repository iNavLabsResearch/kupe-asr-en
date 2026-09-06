"""Make `kupe_asr_en` importable when a script is run as a file path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
