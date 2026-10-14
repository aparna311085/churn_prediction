"""
Wrapper to execute generate_pdf_report from root.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generate_pdf_report import build_pdf

if __name__ == "__main__":
    build_pdf()
