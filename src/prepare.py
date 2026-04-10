import os
import torch
from torchvision import datasets, transforms

# Create output directory
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define transform (convert to tensor only)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download MNIST dataset
train_dataset = datasets.MNIST(
    root="data/raw",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data/raw",
    train=False,
    download=True,
    transform=transform
)

# Convert datasets to tensors
def dataset_to_tensors(dataset):
    images = []
    labels = []

    for img, label in dataset:
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels


train_images, train_labels = dataset_to_tensors(train_dataset)
test_images, test_labels = dataset_to_tensors(test_dataset)

# Save tensors
torch.save((train_images, train_labels), os.path.join(OUTPUT_DIR, "train.pt"))
torch.save((test_images, test_labels), os.path.join(OUTPUT_DIR, "test.pt"))

print("Data preparation complete.")
print(f"Saved to: {OUTPUT_DIR}")