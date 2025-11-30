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

def process_image(image_path):
    """Process a PIL image for use in a PyTorch model. Returns a NumPy array."""
    
    # Load the image
    img = Image.open(image_path)
    
    # Resize: shortest side = 256, maintain aspect ratio
    if img.width < img.height:
        img.thumbnail((256, 256**10))
    else:
        img.thumbnail((256**10, 256))
    
    # Center crop to 224x224
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy and scale to 0-1
    np_image = np.array(img) / 255.0
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to C x H x W
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5, device='cpu'):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    
    # Move model to correct device
    model.to(device)
    model.eval()
    
    # Process the image → numpy array
    np_image = process_image(image_path)
    
    # Convert to torch tensor, add batch dimension
    img_tensor = torch.from_numpy(np_image).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert log probabilities to probabilities
    ps = torch.exp(output)
    
    # Extract top-k probabilities and indices
    top_probs, top_indices = ps.topk(topk, dim=1)
    
    # Move to CPU & flatten
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()
    
    # Invert class_to_idx dict to map predicted index → original class label
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes



def main():

    
    # Parse command-line arguments
   
    args = parser.parse_args()

    image_path = args.image_path
    checkpoint_path = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    use_gpu = args.gpu

    
    # Device selection
    
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    
    # Load category names (JSON)
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)


    # Load model checkpoint
    
    model = load_checkpoint(checkpoint_path, device=device)


    # Make prediction
   
    probs, classes = predict(image_path, model, top_k, device)

    # Convert class labels → flower names
    flower_names = [cat_to_name[c] for c in classes]

 
    # Print results
    
    print("\nTop K Probabilities:")
    print(probs)

    print("\nCorresponding Class Labels:")
    print(classes)

    print("\nFlower Names:")
    for i in range(len(flower_names)):
        print(f"{i+1}: {flower_names[i]} ({probs[i]:.4f})")


if __name__ == "__main__":
    main()
