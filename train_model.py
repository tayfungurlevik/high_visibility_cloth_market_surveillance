import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os

class high_vis_cloth_detector_model(nn.Module):
    def __init__(self, len_classes: int) -> None:
        super().__init__()
        self.feature_extractor1 = nn.Sequential(nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=0),
                                                nn.ReLU())
        self.feature_extractor2 = nn.Sequential(nn.Conv2d(40, 50, kernel_size=3, stride=1, padding=0), nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=3, stride=1, padding=0))
        self.feature_extractor3 = nn.Sequential(nn.Conv2d(50, 40, kernel_size=3, stride=1, padding=0),
                                                nn.ReLU())
        self.feature_extractor4 = nn.Sequential(nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=0), nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=3, stride=1, padding=0))
        
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features=898880, out_features=len_classes))

    def forward(self, x):
        x1 = self.feature_extractor1(x)
        x2 = self.feature_extractor2(x1)
        x3 = self.feature_extractor3(x2)
        x4 = self.feature_extractor4(x3)
        x5 = self.classifier(x4)
        return x5


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
data_folder = "data"

data_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                       transforms.GaussianBlur(kernel_size=3), transforms.ElasticTransform(),
                                       transforms.Resize((224, 224)), transforms.ToTensor()])
ds = ImageFolder(root=data_folder, transform=data_transformer)

train_ds, test_ds = train_test_split(ds, test_size=0.2)
train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=8, shuffle=False)
model = high_vis_cloth_detector_model(len(ds.classes))
model.to(device)
num_epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6)
train_loss = []
test_loss = []
train_acc = []
test_acc = []
best_test_accuracy = 0.0
# Create models directory if it doesn't exist
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# training loop
for epoch in range(num_epochs):
    model.train()
    loss_per_epoch_train = 0.0
    loss_per_epoch_test = 0.0
    accuracy_per_epoch_train = 0.0
    accuracy_per_epoch_test = 0.0
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)  # Move data to GPU
        y_pred_logits = model(x)
        # calculate loss
        loss = loss_fn(y_pred_logits, y)
        loss_per_epoch_train += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy_per_epoch_train += accuracy_fn(y_true=y, y_pred=y_pred_logits.argmax(dim=1))

    loss_per_epoch_train /= len(train_dataloader)
    accuracy_per_epoch_train /= len(train_dataloader)
    train_loss.append(loss_per_epoch_train.cpu().detach().numpy())  # Move loss to CPU before converting to numpy
    train_acc.append(accuracy_per_epoch_train)
    model.eval()
    with torch.inference_mode():
        for batch, (x_test, y_test) in enumerate(test_dataloader):
            x_test, y_test = x_test.to(device), y_test.to(device)  # Move data to GPU
            y_pred_logits_test = model(x_test)
            loss_test = loss_fn(y_pred_logits_test, y_test)
            loss_per_epoch_test += loss_test
            accuracy_per_epoch_test += accuracy_fn(y_true=y_test, y_pred=y_pred_logits_test.argmax(dim=1))
        loss_per_epoch_test /= len(test_dataloader)
        accuracy_per_epoch_test /= len(test_dataloader)
        test_loss.append(loss_per_epoch_test.cpu().detach().numpy())  # Move loss to CPU before converting to numpy
        test_acc.append(accuracy_per_epoch_test)
        
        # Save the best model
        if accuracy_per_epoch_test > best_test_accuracy:
            best_test_accuracy = accuracy_per_epoch_test
            torch.save(model.state_dict(), os.path.join(models_dir, "best.pth"))
            print(f"Model saved with test accuracy: {best_test_accuracy:.2f}%")

    if epoch % 5 == 0:
        print(f"===============Epoch:{epoch}===============")
        print(f"Train Loss:{loss_per_epoch_train:.6f}")
        print(f"Train Accuracy:{accuracy_per_epoch_train:.2f}%")
        print(f"Test Loss:{loss_per_epoch_test:.6f}")
        print(f"Test Accuracy:{accuracy_per_epoch_test:.2f}%")
