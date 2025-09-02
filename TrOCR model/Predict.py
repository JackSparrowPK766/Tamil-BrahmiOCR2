from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt

# === Load model and processor ===
SAVE_DIR = "/content/drive/MyDrive/trocr_brahmi_finetuned"
processor = TrOCRProcessor.from_pretrained(SAVE_DIR)
model = VisionEncoderDecoderModel.from_pretrained(SAVE_DIR)

# === Fix config ===
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Predict function ===
def predict_brahmi_label(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_length=4,
        num_beams=1,
        early_stopping=True
    )

    pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return pred_str.strip()
label_map = {
    0: 'க', 1: 'ர', 2: 'ற', 3: 'ன', 4: 'ஞ', 5: 'ந', 6: 'ஆ', 7: 'வ', 8: 'ல',
    9: 'ண', 10: 'த', 11: 'ப', 12: 'ம', 13: 'பி', 14: 'பூ', 15: 'தி',
    16: 'ழ', 17: 'து', 18: 'ய', 19: 'யு', 20: 'னு', 21: 'நி',
    22: 'னூ', 23: 'எ', 24: 'உ', 25: 'ஓ', 26: 'வூ',27:'', 28:'ெ', 29:'கு', 30:'ஐ', 31:'ஒ', 32:'ரி', 33:'யி', 34:'ளு', 35:'லி', 36:'ட', 37:'ரு', 38:'லு',39:'ள',40:'னி',41:'ஜி'}

# === Predict on an image ===
test_image_path = "/content/v6.jpeg"
plt.imshow(cv2.imread(test_image_path))
predicted_label = predict_brahmi_label(test_image_path)
print("📌 Predicted Subfolder Name: ",label_map[int(predicted_label[:2])])
