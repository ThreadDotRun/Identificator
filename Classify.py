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
        
        print(f"Available classes: {self.classes}")
        
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


# Example usage
model_path = "openai/clip-vit-base-patch32"
classifier = Classify(model_path)

# Single image prediction
image_path = "/media/tdrsvr/a2c28d32-eac8-4438-b371-ce4a345bbf6d/dev/Identificator/testimages/TheMatrixElon.png"
predicted_class, confidence = classifier.predict(image_path)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

# Batch prediction
test_images = [
    "/media/tdrsvr/a2c28d32-eac8-4438-b371-ce4a345bbf6d/dev/Identificator/testimages/TheMatrixElon.png",
    "/media/tdrsvr/a2c28d32-eac8-4438-b371-ce4a345bbf6d/dev/Identificator/testimages/Screenshot from 2022-06-26 21-14-21.png",
    "/media/tdrsvr/a2c28d32-eac8-4438-b371-ce4a345bbf6d/dev/Identificator/testimages/Screenshot from 2022-05-08 09-33-34.png"
]
results = classifier.predict_batch(test_images)
for path, pred_class, conf in results:
    print(f"Image: {path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {conf:.2%}\n")
    