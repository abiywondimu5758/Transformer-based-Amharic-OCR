import os, csv, numpy as np, cv2

# ── Config ──
x_path = 'train_data/X_trainp_pg_vg.npy'
y_path = 'train_data/y_trainp_pg_vg.npy'

# **NEW**: Path to your mapping_labls file (index→char)
mapping_labls_path = 'train_data/mapping_labls'

output_img_dir = '/home/${USER}/Unified_dataset_test/images'
output_csv     = '/home/${USER}/Unified_dataset_test/labels.csv'

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
            # Remove the single-character check so that multi-word labels are preserved
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

# ── Load with mmap ──
X = np.load(x_path, mmap_mode='r')               # e.g. (337332, 48,128)
Y = np.load(y_path, allow_pickle=True).squeeze()  # each element is an array of ints

# ── Prepare CSV writer ──
first_time = not os.path.exists(output_csv)
with open(output_csv, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename','label','type','source','split'])
    if first_time:
        writer.writeheader()

    # Only process first 100 for test
    for idx in range(100):
        img = X[idx]
        # Normalize to uint8 if needed
        if img.dtype != np.uint8:
            img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)
        # Rotate image 90° clockwise and flip horizontally
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)

        fname = f"adotxt_test_{idx:05d}.png"
        out_path = os.path.join(output_img_dir, fname)
        cv2.imwrite(out_path, img)

        # **CHANGE HERE**: Convert Y[idx] (array of ints) → Amharic string
        text_str = indices_to_amharic_string(Y[idx])

        writer.writerow({
            'filename': fname,
            'label':    text_str,
            'type':     'text-line',
            'source':   'ADOCR',
            'split':    'train'
        })

        if idx % 10 == 0:
            print(f"Processed {idx} → {fname}")

print("✅ Test conversion complete (100 lines).")
