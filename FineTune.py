import os
import datetime
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn, optim

from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

folder = "./images/"

for root, _, files in os.walk(folder):
	for file in files:
		try:
			path = os.path.join(root, file)
			image = read_image(path)
			_ = to_pil_image(image)  # Check if conversion to PIL works
		except Exception as e:
			print(f"Problem with image: {path} -> {e}")


class FineTune:
	def __init__(self, model_name: str):
		"""
		Initializes the FineTune class with a pre-trained model and processor.

		:param model_name: Name of the model to load (e.g., 'openai/clip-vit-base-patch32').
		"""
		self.model_name = model_name
		self.model = CLIPModel.from_pretrained(model_name)
		self.processor = CLIPProcessor.from_pretrained(model_name)

	def fine_tune(self, folder_name: str, epochs: int = 3, batch_size: int = 16, learning_rate: float = 1e-4):
		"""
		Fine-tunes the model on the images in the specified folder.
		:param folder_name: Path to the folder containing images for training.
		:param epochs: Number of training epochs (default is 3).
		:param batch_size: Batch size for training (default is 16).
		:param learning_rate: Learning rate for optimizer (default is 1e-4).
		"""
		# Define transforms
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
		])

		# Load dataset
		dataset = datasets.ImageFolder(folder_name, transform=transform)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		# Print class mapping
		class_to_idx = dataset.class_to_idx
		print("\nClass to index mapping:")
		for class_name, idx in class_to_idx.items():
			print(f"Class '{class_name}' -> Index {idx}")

		# Print dataset statistics
		print(f"\nTotal number of samples: {len(dataset)}")
		for class_name in dataset.classes:
			class_idx = class_to_idx[class_name]
			samples_in_class = sum(1 for _, label in dataset if label == class_idx)
			print(f"Samples in '{class_name}': {samples_in_class}")

		# Create text labels for each class
		text_inputs = [f"This is a {class_name} image" for class_name in dataset.classes]
		print(f"\nText labels: {text_inputs}")

		# Sample inspection
		print("\nSample inspection:")
		for idx, (image, label) in enumerate(dataset):
			print(f"Image {idx} - Label: {label} ({dataset.classes[label]})")
			if idx == 5:  # Only show first 6 samples
				break

		# Prepare model for fine-tuning
		self.model.train()
		optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
		loss_fn = nn.CrossEntropyLoss()

		# Training loop
		for epoch in range(epochs):
			total_loss = 0
			for batch_idx, (images, labels) in enumerate(dataloader):
				# Print pixel value range for first batch
				if batch_idx == 0:
					print(f"\nPixel value range: Min={images.min().item():.4f}, Max={images.max().item():.4f}")

				optimizer.zero_grad()

				# Process both images and text
				inputs = self.processor(
					images=[img.permute(1, 2, 0).numpy() for img in images],
					text=text_inputs,
					return_tensors="pt",
					padding=True,
					do_rescale=False  # Images are already scaled
				)

				# Forward pass
				outputs = self.model(**inputs)
				logits = outputs.logits_per_image

				# Compute loss
				loss = loss_fn(logits, labels)
				loss.backward()
				optimizer.step()

				total_loss += loss.item()

				# Print batch progress
				if batch_idx % 5 == 0:
					print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

			# Print epoch summary
			avg_loss = total_loss / len(dataloader)
			print(f"\nEpoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

		# Save the fine-tuned model
		timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		save_name = f"{self.model_name.replace('/', '-')}_{timestamp}_{Path(folder_name).name}"
		self.model.save_pretrained(save_name)
		print(f"\nModel saved as: {save_name}")

	
# Example usage
finetuner = FineTune('openai/clip-vit-base-patch32')
finetuner.fine_tune('./images/')
