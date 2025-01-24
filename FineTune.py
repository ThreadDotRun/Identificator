class FineTune:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def fine_tune(self, folder_name: str, epochs: int = 3, batch_size: int = 16, learning_rate: float = 1e-4):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = datasets.ImageFolder(folder_name, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        class_to_idx = dataset.class_to_idx

        text_inputs = [f"This is a {class_name} image" for class_name in dataset.classes]

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, labels) in enumerate(dataloader):
                optimizer.zero_grad()

                inputs = self.processor(
                    images=[img.permute(1, 2, 0).numpy() for img in images],
                    text=text_inputs,
                    return_tensors="pt",
                    padding=True,
                    do_rescale=False
                )

                outputs = self.model(**inputs)
                logits = outputs.logits_per_image

                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            loss_history.append({'epoch': epoch + 1, 'value': avg_loss})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{self.model_name.replace('/', '-')}_{timestamp}_{Path(folder_name).name}"
        self.model.save_pretrained(save_name)

        return loss_history  # Return loss history
