import os
import numpy as np
from PIL import Image
from datasets import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# === Load Pretrained Model and Processor ===

model_name = "microsoft/trocr-small-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
# Freeze all encoder and decoder layers (fine-tune only lm_head)
for name, param in model.named_parameters():
    if "lm_head" not in name:
        param.requires_grad = False

# === Load Your Dataset ===
BASE = os.path.expanduser('~/gcs_mount')
X_train = np.load(os.path.join(BASE, 'train', 'X_trainp_pg_vg.npy'))
Y_train = np.load(os.path.join(BASE, 'train', 'y_trainp_pg_vg.npy'), allow_pickle=True)
mapping_path = os.path.join(BASE, 'train', 'mapping_labls')

# === Decode Labels ===
idx2char = {}
with open(mapping_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            idx2char[int(parts[0])] = parts[1]

def decode_indices(seq):
    return "".join(idx2char.get(int(idx), "") for idx in seq)

images = [Image.fromarray((img * 255).astype(np.uint8)) for img in X_train]
texts = [decode_indices(seq) for seq in Y_train]

# === Build Dataset ===
dataset = Dataset.from_dict({"image": images, "text": texts})

def preprocess(example):
    encoding = processor(
        images=example["image"],
        text=example["text"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encoding["labels"][encoding["labels"] == processor.tokenizer.pad_token_id] = -100
    return {
        "pixel_values": encoding["pixel_values"].squeeze(),
        "labels": encoding["labels"].squeeze(),
    }

dataset = dataset.map(preprocess)

# === Training Arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=False  # Disable FP16 since you're using CPU
)

# === Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

# === Train the Model ===
trainer.train()

# === Save Final Model ===
model.save_pretrained("./trocr_finetuned")
processor.save_pretrained("./trocr_finetuned")
# === Save Final Model ===
model.save_pretrained("./trocr_finetuned")
processor.save_pretrained("./trocr_finetuned")
