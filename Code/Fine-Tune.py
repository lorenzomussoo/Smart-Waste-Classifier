import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from Train import TrashDataset, TrashCNN
from sklearn.linear_model import SGDClassifier
from torch.nn.modules.loss import _WeightedLoss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(_WeightedLoss):
    def __init__(self, gamma=1.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# CNN fine-tuning function
def fine_tune(model, train_loader, optimizer, criterion, device, num_epochs=10):
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
    
    torch.save(model.state_dict(), '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Trash classifier finetuned.pth')
    np.savez('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Fine-tune metrics.npz',
             loss=loss_values, accuracy=accuracy_values)
    print("Saved CNN fine-tuned model as 'Trash classifier finetuned.pth'")
    print("Saved CNN fine tuning metrics as 'Fine-tune metrics.npz'")

# CNN feature extraction function
def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            x = model.pool(F.relu(model.conv1(images)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            x = model.pool(F.relu(model.conv4(x)))
            x = x.view(x.size(0), -1)
            all_features.append(x.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    return all_features, all_labels

def train_lrsgd(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Scaler.pth')

    X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels, test_size=0.2, stratify=labels)

    lrsgd_model = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-3,
        alpha=1e-4,
        learning_rate="adaptive",
        eta0=0.01,
        warm_start=True
    )

    n_epochs = 10
    loss_values = []
    accuracy_values = []

    for epoch in range(n_epochs):
        lrsgd_model.fit(X_train, y_train)
        preds = lrsgd_model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        try:
            probs = lrsgd_model.predict_proba(X_val)
            if np.isnan(probs).any():
                raise ValueError("NaN in probabilities")
            loss = log_loss(y_val, probs, labels=np.unique(labels))
        except Exception as e:
            loss = np.nan

        accuracy_values.append(acc)
        loss_values.append(loss)
        print(f"Epoch {epoch+1}: val_acc={acc:.4f}, val_loss={loss:.6f}")

    joblib.dump(lrsgd_model, '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Lrsgd model.pth')
    np.savez('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Lrsgd metrics.npz',
             loss=np.array(loss_values),
             accuracy=np.array(accuracy_values))
    print("Saved lrsgd metrics (validation loss + accuracy per epoch) as 'Lrsgd metrics.npz'")

# Main fine-tuning execution
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
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, _ = random_split(dataset, [train_size, test_size])

    class_counts = np.bincount(dataset.labels)
    class_weights_sampler = 1. / class_counts
    train_labels = [dataset.labels[i] for i in train_ds.indices]
    class_counts = np.bincount(train_labels)
    class_weights_sampler = 1. / class_counts
    samples_weight = [class_weights_sampler[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset.labels), y=dataset.labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = TrashCNN(num_classes=len(dataset.classes)).to(device)
    model.load_state_dict(torch.load('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Trash classifier.pth', map_location=device))
    
    criterion = FocalLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # learning rate reduced for fine-tuning

    # CNN fine-tuning
    fine_tune(model, train_loader, optimizer, criterion, device)

    # LR-SGD training
    model_finetuned = TrashCNN(num_classes=len(dataset.classes)).to(device)
    model_finetuned.load_state_dict(torch.load('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Trash classifier finetuned.pth', map_location=device))
    features, labels = extract_features(model, train_loader, device)
    train_lrsgd(features, labels)