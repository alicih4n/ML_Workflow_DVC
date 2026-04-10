import json

import torch
import torch.nn as nn

MODEL_PATH = "model.pt"
TEST_DATA_PATH = "data/processed/test.pt"
PREDICTIONS_PATH = "predictions.json"


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


device = get_device()
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
if "model_state_dict" in checkpoint:
    activation_name = checkpoint.get("activation", "relu")
    state_dict = checkpoint["model_state_dict"]
else:
    activation_name = "relu"
    state_dict = checkpoint

model = SimpleCNN(activation_name).to(device)
model.load_state_dict(state_dict)
model.eval()

test_images, test_labels = torch.load(TEST_DATA_PATH, map_location="cpu")
test_images = test_images.float().to(device)
test_labels = test_labels.long()

with torch.no_grad():
    outputs = model(test_images)
    predicted = outputs.argmax(dim=1).cpu()

predictions = []
for index in range(10):
    predictions.append(
        {
            "index": index,
            "true_label": int(test_labels[index].item()),
            "predicted_label": int(predicted[index].item()),
        }
    )

with open(PREDICTIONS_PATH, "w", encoding="utf-8") as file:
    json.dump(predictions, file, indent=4)

print(f"Saved 10 predictions to {PREDICTIONS_PATH}")
