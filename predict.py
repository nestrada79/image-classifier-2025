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

def main():
    pass

if __name__ == "__main__":
    main()
