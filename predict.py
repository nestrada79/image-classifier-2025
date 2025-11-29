import argparse
import json
import torch
from torchvision import models
import numpy as np
from PIL import Image
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Flower Image Classifier Prediction Script")

parser.add_argument("image_path", 
                    help="Path to the input image for prediction")

parser.add_argument("checkpoint", 
                    help="Path to the trained model checkpoint")

parser.add_argument("--top_k", 
                    type=int, 
                    default=5, 
                    help="Return top K most likely classes")

parser.add_argument("--category_names", 
                    default="cat_to_name.json",
                    help="Path to JSON file mapping categories to flower names")

parser.add_argument("--gpu", 
                    action="store_true",
                    help="Use GPU for inference if available")

def load_checkpoint(filepath, device='cpu'):
    """Load a trained model checkpoint for inference."""

    # PyTorch 2.6 requires disabling weights_only for full checkpoint loading
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load the architecture
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {checkpoint['architecture']}")

    # Restore classifier
    model.classifier = checkpoint['classifier']

    # Restore mapping
    model.class_to_idx = checkpoint['class_to_idx']

    # Restore weights
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model.eval()

    return model


def main():
    pass

if __name__ == "__main__":
    main()
