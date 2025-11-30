import argparse
import json
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import os

# Argument Parser

parser = argparse.ArgumentParser(description="Train a Flower Image Classifier")

# Positional argument: dataset directory (must contain train/valid/test)
parser.add_argument('data_dir', 
                    help='Directory containing the dataset')

# Optional arguments
parser.add_argument('--save_dir', 
                    type=str, 
                    default='.', 
                    help='Directory to save the trained checkpoint')

parser.add_argument('--arch', 
                    type=str, 
                    default='vgg16', 
                    help='Model architecture (vgg16 supported)')

parser.add_argument('--learning_rate', 
                    type=float, 
                    default=0.003, 
                    help='Learning rate')

parser.add_argument('--hidden_units', 
                    type=int, 
                    default=512, 
                    help='Number of hidden units in the classifier')

parser.add_argument('--epochs', 
                    type=int, 
                    default=5, 
                    help='Number of epochs to train')

parser.add_argument('--gpu', 
                    action='store_true', 
                    help='Use GPU for training if available')

# Load and Transform the Data

def load_data(data_dir):
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transforms = valid_transforms

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64)

    return train_loader, valid_loader, test_loader, train_data.class_to_idx

# Build the Model

def build_model(arch='vgg16', hidden_units=512, learning_rate=0.003):
    
    # Load a pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Only vgg16 is supported in this script.")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Get the input size of the classifier
    input_size = model.classifier[0].in_features
    
    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),     # 102 flower categories
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Loss function
    criterion = nn.NLLLoss()

    # Optimizer (only train classifier parameters)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


# Validation Function

def validate(model, dataloader, criterion, device):
    model.eval()
    loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, preds = torch.max(outputs, 1)
            accuracy += torch.mean((preds == labels).float()).item()

    return loss / len(dataloader), accuracy / len(dataloader)


# Train the Model

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    print("Training starting...\n")

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation after each epoch
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        print(f"Epoch {e+1}/{epochs}.. "
              f"Train Loss: {running_loss/len(train_loader):.3f}.. "
              f"Valid Loss: {valid_loss:.3f}.. "
              f"Valid Acc: {valid_acc:.3f}")

    print("\nTraining complete!")
    return model


# Save the Checkpoint

def save_checkpoint(model, save_dir, arch, class_to_idx):
    checkpoint = {
        'architecture': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }

    # Ensure save dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Full path to save file
    save_path = os.path.join(save_dir, 'checkpoint.pth')

    torch.save(checkpoint, save_path)

    print(f"\nCheckpoint saved to: {save_path}")




def main():
    
    pass

if __name__ == "__main__":
    main()
