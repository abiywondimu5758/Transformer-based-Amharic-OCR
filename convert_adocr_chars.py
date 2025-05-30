import os
import numpy as np
import pandas as pd
import cv2


char_train_dir = 'train_char'
char_test_dir  = 'test_char'

output_img_dir = r'D:\Windows Only\OCR-NLP\Datasets\Unified_dataset/images'
output_csv = r'D:\Windows Only\OCR-NLP\Datasets\Unified_dataset/labels.csv'


os.makedirs(output_img_dir, exist_ok=True)

def process_split(split_name, npy_dir, prefix):
    x_path = os.path.join(npy_dir, 'x_' + split_name + '.npy')
    y_path = os.path.join(npy_dir, 'y_' + split_name + '.npy')
    
   
    X = np.load(x_path, mmap_mode='r')
    Y = np.load(y_path, allow_pickle=True).squeeze()
    
    print(f"[{split_name}] Loaded X={X.shape}, Y={Y.shape}, dtype={Y.dtype}")
    
    records = []
    for i in range(len(X)):
        
        fname = f"{prefix}_{split_name}_{i:05d}.png"
        out_path = os.path.join(output_img_dir, fname)
        
        
        img = X[i]
        if img.dtype != np.uint8:
            img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)
        
        
        cv2.imwrite(out_path, img)
        
        
        records.append({
            'filename': fname,
            'label': str(Y[i]),
            'type':    'character',
            'source':  'ADOCR',
            'split':   split_name
        })
    return records


all_records = []
all_records += process_split('train', char_train_dir, 'adochar')
all_records += process_split('test',  char_test_dir,  'adochar')


df = pd.DataFrame(all_records)
df.to_csv(output_csv, index=False)
print("✅ Character dataset conversion complete. Wrote", len(df), "entries to", output_csv)
