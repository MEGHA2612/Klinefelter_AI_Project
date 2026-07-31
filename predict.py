import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class KlinefelterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = KlinefelterCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor()
])

img = Image.open("test.jpeg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)

    probabilities = torch.softmax(output, dim=1)

    confidence, pred = torch.max(probabilities, 1)

if pred.item() == 0:
    print(f"Klinefelter Detected ({confidence.item()*100:.2f}%)")
else:
    print(f"Normal Karyotype ({confidence.item()*100:.2f}%)")