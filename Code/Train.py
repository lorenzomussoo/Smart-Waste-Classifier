import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, random_split

class FocalLoss(_WeightedLoss):
    def __init__(self, gamma=1.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Dataset definition
class TrashDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.img_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.startswith('.') or not os.path.isfile(img_path):
                    continue
                self.img_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# Model definition
class TrashCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrashCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))  
        x = x.view(-1, 128 * 8 * 8) 
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=40):
    model.train()
    loss_values = []
    accuracy_values = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        loss_values.append(epoch_loss)
        accuracy_values.append(epoch_accuracy)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    torch.save(model.state_dict(), '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Trash classifier.pth')
    print("Saved CNN model as 'Trash classifier.pth'")
    np.savez('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Training metrics.npz',
             loss=loss_values, accuracy=accuracy_values)
    print("Saved training metrics as 'Training metrics.npz'")

# Main training execution
if __name__ == "__main__":
    root_dir = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/Data'
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])

    dataset = TrashDataset(root_dir, transform=train_transform)
    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, _ = random_split(dataset, [train_size, test_size])

    train_labels = [train_ds.dataset.labels[i] for i in train_ds.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    samples_weight = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset.labels), y=dataset.labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = TrashCNN(num_classes=len(class_names)).to(device)
    criterion = FocalLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # CNN
    train_model(model, train_loader, optimizer, criterion, device)