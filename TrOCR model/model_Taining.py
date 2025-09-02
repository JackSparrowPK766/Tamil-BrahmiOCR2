import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import Dataset, DatasetDict
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator


DATASET_PATH = '/content/drive/MyDrive/brami1'  # ✅ CHANGE IF NEEDED
IMAGE_SIZE = 64
MAX_LABEL_LENGTH = 4


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


image_paths = []
labels = []

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, file))
                labels.append(label)

print(f"✅ Unique labels (classes): {sorted(set(labels))}")

from collections import defaultdict

# Group images by label
label_to_images = defaultdict(list)
for path, label in zip(image_paths, labels):
    label_to_images[label].append(path)

# Filter: Keep only classes with 2 or more images
filtered_image_paths = []
filtered_labels = []

for label, paths in label_to_images.items():
    if len(paths) >= 2:
        filtered_image_paths.extend(paths)
        filtered_labels.extend([label] * len(paths))

# Sanity check
print(f"✅ Total images after filtering: {len(filtered_image_paths)}")
print(f"✅ Classes retained: {sorted(set(filtered_labels))}")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def encode_example(image_path, label):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
    label_tensor = processor.tokenizer(label, padding="max_length", max_length=MAX_LABEL_LENGTH,
                                       truncation=True, return_tensors="pt").input_ids.squeeze(0)
    label_tensor[label_tensor == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": label_tensor}


# === Install tqdm if not already installed ===
!pip install -q tqdm

# === Imports ===
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from PIL import Image

# === Assume processor and MAX_LABEL_LENGTH are already defined ===

def encode_example(image_path, label):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
    label_tensor = processor.tokenizer(label, padding="max_length", max_length=MAX_LABEL_LENGTH,
                                       truncation=True, return_tensors="pt").input_ids.squeeze(0)
    label_tensor[label_tensor == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": label_tensor}

def build_dataset(paths, labels):
    encodings = [encode_example(img, lbl) for img, lbl in tqdm(zip(paths, labels), total=len(labels), desc="📦 Encoding Images")]
    return Dataset.from_dict({
        "pixel_values": [e["pixel_values"] for e in encodings],
        "labels": [e["labels"] for e in encodings]
    })

# === Build Train and Test datasets with progress ===
train_dataset = build_dataset(train_paths, train_labels)
test_dataset = build_dataset(test_paths, test_labels)

# === Combine into DatasetDict ===
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})


from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback
)
import torch

# ==== Training Arguments ====
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_brahmi",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    num_train_epochs=150,
    save_total_limit=1,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",              # 🔥 Disable external logging (wandb, etc.)
    evaluation_strategy="epoch",   # Required for early stopping
    save_strategy="epoch",         # Save every epoch (so early stopping can monitor)
    logging_strategy="epoch",      # Log every epoch
    load_best_model_at_end=True,   # Load best model after training
    metric_for_best_model="loss",  # Metric to monitor for early stopping
    greater_is_better=False,       # Lower loss is better
    learning_rate=5e-5             # ✅ Add learning rate
)

# ==== Load Model and Processor ====
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# ✅ Required config settings
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.to("cuda" if torch.cuda.is_available() else "cpu")

# ==== Trainer ====
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=processor.tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=default_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]  # ✅ Early stop
)

# ==== Train ====
trainer.train()


preds = trainer.predict(dataset["test"])
pred_strs = processor.batch_decode(preds.predictions, skip_special_tokens=True)
true_strs = processor.batch_decode(preds.label_ids, skip_special_tokens=True)
print(f"Test Accuracy: {accuracy_score(true_strs, pred_strs) * 100:.2f} %")

report = classification_report(true_strs, pred_strs, zero_division=0, output_dict=True)
print(f"Accuracy:  {accuracy_score(true_strs, pred_strs)}")
print(f"Precision: {report['macro avg']['precision']}")
print(f"Recall:    {report['macro avg']['recall']}")
print(f"F1-score:  {report['macro avg']['f1-score']}")
print("📋 Classification Report (Macro Average):")
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# 11. Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy over Epochs'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss over Epochs'); plt.legend()
plt.tight_layout()
plt.show()


