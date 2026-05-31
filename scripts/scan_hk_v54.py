"""v5.4 HK scan entry point for CI workflow."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from run_scan import main as scan_main

if __name__ == '__main__':
    scan_main()
