import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

DATA_DIR = "data/processed"
MODEL_PATH = "model.pt"
METRICS_PATH = "metrics.json"
PARAMS_PATH = "params.yaml"
SEED = 42


def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_activation(name):
    activation_name = name.lower().replace("_", "")
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leakyrelu":
        return nn.LeakyReLU()
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(
        "Unsupported activation. Choose one of: relu, leakyrelu, gelu."
    )


def build_optimizer(parameters, optimizer_name, learning_rate, momentum):
    normalized_name = optimizer_name.lower().replace("-", "_")
    if normalized_name == "adam":
        return optim.Adam(parameters, lr=learning_rate)
    if normalized_name == "sgd":
        return optim.SGD(parameters, lr=learning_rate, momentum=0.0)
    if normalized_name in {"sgd_momentum", "momentum", "sgd_with_momentum"}:
        return optim.SGD(parameters, lr=learning_rate, momentum=momentum)
    raise ValueError(
        "Unsupported optimizer. Choose one of: sgd, sgd_momentum, adam."
    )


class SimpleCNN(nn.Module):
    def __init__(self, activation_name):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.activation = get_activation(activation_name)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(self.activation(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


torch.manual_seed(SEED)

params = load_params()
epochs = int(params["epochs"])
learning_rate = float(params["lr"])
batch_size = int(params["batch_size"])
activation_name = params["activation"]
optimizer_name = params["optimizer"]
momentum = float(params["momentum"])
device = get_device()

train_images, train_labels = torch.load(
    os.path.join(DATA_DIR, "train.pt"), map_location="cpu"
)
test_images, test_labels = torch.load(
    os.path.join(DATA_DIR, "test.pt"), map_location="cpu"
)

train_images = train_images.float()
test_images = test_images.float()
train_labels = train_labels.long()
test_labels = test_labels.long()

model = SimpleCNN(activation_name).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = build_optimizer(model.parameters(), optimizer_name, learning_rate, momentum)

last_loss = None

model.train()
for epoch in range(epochs):
    for batch_start in range(0, len(train_images), batch_size):
        x_batch = train_images[batch_start : batch_start + batch_size].to(device)
        y_batch = train_labels[batch_start : batch_start + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        if epoch == 0 and batch_start == 0:
            print("First batch outputs sample:", outputs[:5].detach().cpu().tolist())
            print("First batch loss:", loss.item())
            print(
                "First batch conv gradient norm:",
                model.conv.weight.grad.norm().item(),
            )

        optimizer.step()
        last_loss = loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {last_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_start in range(0, len(test_images), batch_size):
        x_batch = test_images[batch_start : batch_start + batch_size].to(device)
        y_batch = test_labels[batch_start : batch_start + batch_size].to(device)
        outputs = model(x_batch)
        predicted = outputs.argmax(dim=1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total

metrics = {
    "accuracy": accuracy,
    "loss": last_loss,
}
checkpoint = {
    "model_state_dict": model.state_dict(),
    "activation": activation_name,
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

torch.save(checkpoint, MODEL_PATH)

print(f"Device: {device}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Model saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")
