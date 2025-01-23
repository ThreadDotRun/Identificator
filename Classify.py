import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path

class Classify:
    def __init__(self, model_path: str):
        """
        Initialize the classifier with a fine-tuned model.
        
        :param model_path: Path to the fine-tuned model directory
        """
        # Load fine-tuned model from local path
        self.model = CLIPModel.from_pretrained(model_path)
        # Load processor from original CLIP model
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Get class names from the original dataset path
        dataset_path = Path('./images')  # Adjust this to your dataset path
        self.classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        self.text_inputs = [f"This is a {class_name} image" for class_name in self.classes]
        
        print(f"Available classes: {self.classes}") # This is dynamically assigned from folder names
        
        # Set model to evaluation mode
        self.model.eval()

    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Classify a single image.
        
        :param image_path: Path to the image file
        :return: Tuple of (predicted_class, confidence_score)
        """
        # Load and preprocess the image
        image = Image.open(image_path)
        
        # Process both image and text
        inputs = self.processor(
            images=image,
            text=self.text_inputs,
            return_tensors="pt",
            padding=True,
            do_rescale=False
        )

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get the predicted class and confidence
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        predicted_class = self.classes[pred_idx]
        
        return predicted_class, confidence

    def predict_batch(self, image_paths: list[str]) -> list[tuple[str, float]]:
        """
        Classify multiple images.
        
        :param image_paths: List of paths to image files
        :return: List of tuples (predicted_class, confidence_score)
        """
        results = []
        for image_path in image_paths:
            try:
                prediction = self.predict(image_path)
                results.append((image_path, *prediction))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((image_path, "error", 0.0))
        return results

