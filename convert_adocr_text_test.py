import os, csv, numpy as np, cv2

# ── Config ──
# List all test-split .npy pairs here
test_splits = [
    ('test/x_testp.npy',   'test/y_testp.npy'),
    ('test/x_test_pg.npy', 'test/y_test_pg.npy'),
    ('test/x_test_vg.npy', 'test/y_test_vg.npy'),
]

# Path to your mapping_labls file (index→char)
mapping_labls_path = 'test/mapping_labls'

# Get the home directory from the environment
output_img_dir = r'D:\Windows Only\OCR-NLP\Datasets\Unified_dataset/tt/images'
output_csv = r'D:\Windows Only\OCR-NLP\Datasets\Unified_dataset/tt/labels.csv'

os.makedirs(output_img_dir, exist_ok=True)

# ── Step 0: Build index→character dict from mapping_labls ──
def load_index2char(path):
    index2char = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            label = parts[1].strip()
            # Preserve multi-character labels as provided
            index2char[idx] = label
    return index2char

index2char = load_index2char(mapping_labls_path)
# Assume index 0 = <PAD>, index 1 = <BLANK/EOS>
pad_idx = 0
eos_idx = 1

def indices_to_amharic_string(idx_array):
    """
    Convert a 1D array of integer indices into an Amharic string,
    stopping only at pad_idx and replacing eos_idx with a space.
    """
    chars = []
    for idx in idx_array:
        if idx == pad_idx:  # stop conversion only for pad tokens
            break
        if idx == eos_idx:
            chars.append(" ")
        else:
            chars.append(index2char.get(int(idx), ''))
    return ''.join(chars)

# ── Prepare CSV writer once ──
first_time = not os.path.exists(output_csv)
csv_file = open(output_csv, 'a', newline='', encoding='utf-8')
writer   = csv.DictWriter(csv_file, fieldnames=['filename','label','type','source','split'])
if first_time:
    writer.writeheader()

# ── Loop over each test-split file pair ──
for x_path, y_path in test_splits:
    # Load arrays with mmap (X for images) and allow_pickle (Y for labels)
    X = np.load(x_path, mmap_mode='r')               # e.g. (N, 48, 128)
    Y = np.load(y_path, allow_pickle=True).squeeze()  # each element is an array of ints
    N = len(X)
    split_name = os.path.basename(x_path).replace('.npy','')  # e.g. "x_testp", "x_test_pg", "x_test_vg"
    print(f"Converting {split_name} ({N} images)...")

    for idx in range(N):
        img = X[idx]
        # Normalize to uint8 if needed
        if img.dtype != np.uint8:
            img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)
        # Rotate image 90° clockwise and flip horizontally
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)

        fname = f"adotxt_{split_name}_{idx:05d}.png"
        out_path = os.path.join(output_img_dir, fname)
        cv2.imwrite(out_path, img)

        # Convert Y[idx] (array of ints) → Amharic string
        text_str = indices_to_amharic_string(Y[idx])

        writer.writerow({
            'filename': fname,
            'label':    text_str,
            'type':     'text-line',
            'source':   'ADOCR',
            'split':    'test'
        })

        if idx % 50000 == 0:
            print(f"Processed {split_name} {idx} → {fname}")

    print(f"Finished converting {split_name}.")

csv_file.close()
print("✅ All test-split text-lines converted to PNG + CSV.")
