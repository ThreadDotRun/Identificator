# ML Image Categorization Model and Trainer w/ Web GUI Interface

## Overview
A Quart-based web application for machine learning model interactions, specifically designed for image-based fine-tuning and classification tasks. Using this app you can fine tune open models on your images (in folders by category) and then infer new images against these new categories.

<video width="320" height="240" controls>
  <source src="./Identificator.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Features
- Fine-tune machine learning models using image datasets
- Classify images through a web interface
- Supports ZIP file uploads for model training
- Simple, responsive web UI

## Prerequisites
- Python 3.8+
- pip package manager

## Installation
1. Clone the repository
```bash
git clone https://github.com/ThreadDotRun/Identificator.git
cd Identificator
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python Discriminator.py
```
Access the interface at `http://localhost:5000`

## Endpoints
- `/`: Home page with upload forms
- `/finetune`: Upload and process ZIP files for model fine-tuning
- `/classify`: Upload images for classification

## Configuration
- Supports ZIP, PNG, JPG, JPEG file uploads
- Uploads saved in `./uploads` directory

## TODO
- Implement actual fine-tuning logic
- Complete image classification process
- Add error handling and validation

## License
- [Apeche 2.0]

## Contributing
- Pull requests are welcome. For major changes, please open an issue first.
- This code is in pre-alpha ATM so do not expect it to be functional yet.