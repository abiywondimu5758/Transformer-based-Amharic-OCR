import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import editdistance
import sys
import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass  # Needed for Python 3 compatibility

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

def compute_cer(pred_texts, true_texts):
    total_edits = 0
    total_chars = 0
    for pred, true in zip(pred_texts, true_texts):
        total_edits += editdistance.eval(pred, true)
        total_chars += len(true)
    return total_edits / total_chars if total_chars > 0 else 0

def compute_wer(pred_texts, true_texts):
    total_edits = 0
    total_words = 0
    for pred, true in zip(pred_texts, true_texts):
        pred_words = pred.split()
        true_words = true.split()
        total_edits += editdistance.eval(pred_words, true_words)
        total_words += len(true_words)
    return total_edits / total_words if total_words > 0 else 0

def compute_char_accuracy(pred_texts, true_texts):
    correct = 0
    total = 0
    for pred, true in zip(pred_texts, true_texts):
        for p, t in zip(pred, true):
            if p == t:
                correct += 1
        total += len(true)
    return correct / total if total > 0 else 0

# ============ Paths ============
BASE = os.path.expanduser('~/gcs_mount')
X_train_path = os.path.join(BASE, 'train', 'X_trainp_pg_vg.npy')
Y_train_path = os.path.join(BASE, 'train', 'y_trainp_pg_vg.npy')
X_test_path  = os.path.join(BASE, 'test',  'x_testp.npy')
Y_test_path  = os.path.join(BASE, 'test',  'y_testp.npy')
mapping_path = os.path.join(BASE, 'train', 'mapping_labls')

# ============ Load mapping ============
def load_index2char(path):
    idx2char = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx2char[int(parts[0])] = parts[1]
    return idx2char

idx2char = load_index2char(mapping_path)

# ============ Decode function ============
def decode_indices(arr, pad_idx=0, eos_idx=1):
    chars = []
    for idx in arr:
        if idx == pad_idx:
            break
        if idx == eos_idx:
            chars.append(' ')
        else:
            chars.append(idx2char.get(int(idx), ''))
    return ''.join(chars)

# ============ Load raw arrays ============
X_train = np.load(X_train_path, mmap_mode='r')
Y_train = np.load(Y_train_path, allow_pickle=True).squeeze()
X_test  = np.load(X_test_path,  mmap_mode='r')
Y_test  = np.load(Y_test_path,  allow_pickle=True).squeeze()

# ============ Prepare labels ============
train_texts   = [decode_indices(y) for y in Y_train]
test_texts    = [decode_indices(y) for y in Y_test]
le            = LabelEncoder().fit(list(''.join(train_texts)))
num_classes   = len(le.classes_) + 1   # extra class for CTC blank
blank_idx     = num_classes - 1        # index of the CTC-blank token
max_len       = max(len(t) for t in train_texts)
batch_size    = 64

# ============ Generator ============
def gen(X, Y):
    for x, y in zip(X, Y):
        img = x.astype('float32') / 255.0
        img = np.expand_dims(img, -1)
        if img.shape == (128, 48, 1):
            img = np.transpose(img, (1, 0, 2))  # fix incorrect shape
        text = decode_indices(y)
        seq  = le.transform(list(text))
        # pad with the CTC-blank index, not -1
        label = np.pad(seq,
                       (0, max_len - len(seq)),
                       constant_values=blank_idx)
        input_len = np.array([32], dtype=np.int32)
        label_len = np.array([len(seq)], dtype=np.int32)
        yield img, label, input_len, label_len


# ============ Dataset Builder ============
def make_dataset(X, Y, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: gen(X, Y),
        output_signature=(
            tf.TensorSpec(shape=(48,128,1), dtype=tf.float32),
            tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
# ─── Subsample for quick debug ───
DEBUG_SAMPLES = 1000
X_train = X_train[:DEBUG_SAMPLES]
Y_train = Y_train[:DEBUG_SAMPLES]

# Build the small‐subset training dataset
train_ds = make_dataset(X_train, Y_train, shuffle=True)
# Validation still uses full test set
val_ds   = make_dataset(X_test,  Y_test)

# Recompute steps_per_epoch for 1000 samples
steps_per_epoch = int(np.ceil(DEBUG_SAMPLES / batch_size))


# ============ Build CRNN+CTC Model ============
import tensorflow.keras.layers as L

def build_crnn():
    inp = L.Input((48,128,1), name='img')
    x = L.Conv2D(32,3,padding='same',activation='relu')(inp)
    x = L.MaxPool2D((2,2))(x)
    x = L.Conv2D(64,3,padding='same',activation='relu')(x)
    x = L.MaxPool2D((2,2))(x)
    x = L.Reshape((32, 64*12))(x)
    x = L.Bidirectional(L.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = L.Bidirectional(L.LSTM(256, return_sequences=True, dropout=0.3))(x)
    outputs = L.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inp, outputs=outputs)

model = build_crnn()
model.summary()

# ============ CTC Loss & Optimizer ============
optimizer = tf.keras.optimizers.Adam()

@tf.function
def compute_loss(labels, y_pred, input_len, label_len):
    loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)
    return tf.reduce_mean(loss)

steps_per_epoch = int(len(X_train) / batch_size)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=steps_per_epoch * 10,
    alpha=0.1
)
optimizer.learning_rate = lr_schedule

# ============ Training Loop ============

epochs = 10
patience = 2
best_val = np.inf
wait = 0

for epoch in range(1, epochs+1):
    total_loss = 0
    for batch in tqdm(train_ds, desc=f"Epoch {epoch}/{epochs}"):
        img, label, il, ll = batch
        with tf.GradientTape() as tape:
            preds = model(img, training=True)
            loss = compute_loss(label, preds, il, ll)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        total_loss += loss
    train_loss = total_loss / steps_per_epoch

    val_loss = 0
    val_steps = int(len(X_test) / batch_size)
    decoded_preds = []
    decoded_labels = []

    # Updated validation loop with batch_idx enumeration and debug prints
    for batch_idx, (img, label, il, ll) in enumerate(val_ds):
        preds = model(img, training=False)
        val_loss += compute_loss(label, preds, il, ll)

        # Debug print for first batch
        if batch_idx == 0:
            seq_len = tf.reshape(il, [-1])
            decoded_batch, _ = tf.keras.backend.ctc_decode(preds, input_length=seq_len)
            decoded_batch = decoded_batch[0].numpy()
            label_np = label.numpy()
            print("\n Sample predictions:")
            for p_seq, t_seq in zip(decoded_batch[:5], label_np[:5]):
                # filter out blank_idx
                valid_p = [i for i in p_seq if 0 <= i < len(le.classes_)]
                valid_t = [i for i in t_seq if 0 <= i < len(le.classes_)]
                p_text = "".join(le.inverse_transform(valid_p))
                t_text = "".join(le.inverse_transform(valid_t))
                print(f"  GT  : {t_text}")
                print(f"  PRED: {p_text}")
                print("  " + "-" * 30)


        # ─── Fix: reshape il to a 1-D vector of length batch_size ───
        seq_len = tf.reshape(il, [-1])

        # Decode with the corrected input_length
        decoded_batch, _ = tf.keras.backend.ctc_decode(preds, input_length=seq_len)
        decoded_batch = decoded_batch[0].numpy()
        label_np = label.numpy()

    for pred_seq, true_seq in zip(decoded_batch, label_np):
        valid_p = [i for i in pred_seq if 0 <= i < len(le.classes_)]
        valid_t = [i for i in true_seq if 0 <= i < len(le.classes_)]
        pred_text = "".join(le.inverse_transform(valid_p))
        true_text = "".join(le.inverse_transform(valid_t))
        decoded_preds.append(pred_text)
        decoded_labels.append(true_text)

    val_loss /= val_steps
    cer = compute_cer(decoded_preds, decoded_labels)
    wer = compute_wer(decoded_preds, decoded_labels)
    char_acc = compute_char_accuracy(decoded_preds, decoded_labels)

    print(f"Epoch {epoch}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"CER: {cer:.4f}, WER: {wer:.4f}, Char Acc: {char_acc:.4f}")


    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        model.save('best_crnn_ctc.h5')
        model.save('best_crnn_ctc.keras')
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping. Restoring best model.")
            model = tf.keras.models.load_model('best_crnn_ctc.keras', compile=False)
            break

# ============ Save Final Model ============
model.save('crnn_ctc_full_enhanced.keras')
model.save('crnn_ctc_full_enhanced.h5')