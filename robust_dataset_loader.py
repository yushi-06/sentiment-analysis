#!/usr/bin/env python3
"""
Robust Dataset Loader
Handles various file formats and encoding issues automatically
"""

import pandas as pd
import os
from pathlib import Path

class RobustDatasetLoader:
    def __init__(self):
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        self.supported_extensions = ['.csv', '.xlsx', '.xls', '.tsv', '.txt']
    
    def load_dataset(self, file_path):
        """Load dataset with automatic format and encoding detection"""
        print(f"Loading dataset: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.xlsx', '.xls']:
            return self._load_excel(file_path)
        elif file_ext in ['.csv', '.tsv', '.txt']:
            return self._load_csv(file_path, file_ext)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_excel(self, file_path):
        """Load Excel file with error handling"""
        try:
            df = pd.read_excel(file_path)
            print(f"Successfully loaded Excel file")
            return df
        except ImportError:
            print("Missing openpyxl dependency. Installing...")
            import subprocess
            subprocess.run(['pip', 'install', 'openpyxl'], check=True)
            df = pd.read_excel(file_path)
            print(f"Successfully loaded Excel file after installing dependency")
            return df
        except Exception as e:
            raise Exception(f"Error loading Excel file: {e}")
    
    def _load_csv(self, file_path, file_ext):
        """Load CSV/TSV file with encoding detection"""
        separator = '\t' if file_ext == '.tsv' else ','
        
        # Try different encodings
        for encoding in self.supported_encodings:
            try:
                print(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, sep=separator)
                print(f"Successfully loaded with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError as e:
                print(f"Parser warning with {encoding}: {e}")
                # Try with different separator or error handling
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=separator, on_bad_lines='skip')
                    print(f"Loaded with {encoding} (skipped bad lines)")
                    return df
                except:
                    continue
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                continue
        
        # If all encodings fail, try with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', sep=separator)
            print(f"Loaded with UTF-8 (ignored errors)")
            return df
        except Exception as e:
            raise Exception(f"Could not load file with any encoding: {e}")
    
    def save_as_utf8_csv(self, df, output_path):
        """Save dataframe as UTF-8 CSV"""
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved as UTF-8 CSV: {output_path}")
        return output_path

def test_robust_loader():
    """Test the robust loader"""
    loader = RobustDatasetLoader()
    
    # Test files
    test_files = [
        "fix_bias_data.csv",
        "my_reviews.csv",
        "sample_movie_reviews.csv"
    ]
    
    print("TESTING ROBUST DATASET LOADER")
    print("="*45)
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                print(f"\nTesting: {file_path}")
                df = loader.load_dataset(file_path)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"Success!")
            except Exception as e:
                print(f"Failed: {e}")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    test_robust_loader()
