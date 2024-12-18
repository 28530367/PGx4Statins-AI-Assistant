import os
import pandas as pd

def convert_txt_to_csv(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for file in files:
        txt_path = os.path.join(folder_path, file)
        csv_path = os.path.join(folder_path, file.replace('.txt', '.csv'))
        
        try:
            df = pd.read_csv(txt_path, sep='\t', encoding='utf-8')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Success: {file} -> {csv_path}")
        except Exception as e:
            print(f"failed: {file}, error: {e}")

folder_path = "/media/disk2/HSW/PGx4Statins-AI-Assistant/data/test_queries/subsets_txt"
convert_txt_to_csv(folder_path)