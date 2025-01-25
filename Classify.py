import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path

class Classify:
	def __init__(self, model_path: str):
		self.model = CLIPModel.from_pretrained(model_path)
		self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
		self.model.eval()

	def predict(self, image_path: str, classes: list[str]) -> tuple[str, float]:
		"""Classify image using provided class list"""
		image = Image.open(image_path)
		
		# Create natural language prompts
		text_inputs = [f"A photo of a {class_name}" for class_name in classes]
		
		# Process with proper normalization
		inputs = self.processor(
			images=image,
			text=text_inputs,
			return_tensors="pt",
			padding=True
		)

		# Get predictions
		with torch.no_grad():
			outputs = self.model(**inputs)
			logits = outputs.logits_per_image[0]
			probs = torch.nn.functional.softmax(logits, dim=0)
		
		# Get results
		pred_idx = torch.argmax(probs).item()
		return classes[pred_idx], round(probs[pred_idx].item(), 4)

	# predict_batch remains unchanged
	def predict_batch(self, image_paths: list[str], class_list: list[str]) -> list[tuple]:
		results = []
		for image_path in image_paths:
			try:
				predicted_class, confidence = self.predict(image_path, class_list)
				results.append((image_path, predicted_class, confidence))
			except Exception as e:
				print(f"Error processing {image_path}: {e}")
				results.append((image_path, "error", 0.0))
		return results
