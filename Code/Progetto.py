import os
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------
# Dataset Definition
# ------------------------

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------
# Transformations
# ------------------------

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

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ------------------------
# CNN Model
# ------------------------

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

# ------------------------
# Feature Extraction for LR-SGD
# ------------------------

def extract_features_for_lrsgd(model, image_tensor):
    with torch.no_grad():
        x = model.pool(F.relu(model.conv1(image_tensor)))
        x = model.pool(F.relu(model.conv2(x)))
        x = model.pool(F.relu(model.conv3(x)))
        x = model.pool(F.relu(model.conv4(x)))
        x = x.view(-1, 128 * 8 * 8)
    return x.cpu().numpy()

# ------------------------
# Evaluation
# ------------------------

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

# ------------------------
# Confusion Matrix
# ------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    save_plot_to_pdf(title, plt.gcf())

# ------------------------
# Training Metrics Plot
# ------------------------

def plot_training_metrics(metric_path, model_name="Training"):
    data = np.load(metric_path)
    loss = data['loss']
    accuracy = data['accuracy']
    epochs = range(1, len(loss)+1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, 'b-o', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss per Epoch')
    plt.legend()
    save_plot_to_pdf(f'{model_name} Loss', plt.gcf())
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, 'g-o', label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy per Epoch')
    plt.legend()
    save_plot_to_pdf(f'{model_name} Accuracy', plt.gcf())
    plt.close()

# ------------------------
# Learning Curve Plot
# ------------------------

def plot_learning_curve_cnn(model_class, dataset, device, class_names, title="CNN Learning Curve"):
    from sklearn.metrics import accuracy_score
    train_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    train_accuracies = []
    val_accuracies = []
    for frac in train_fractions:
        # Split dataset
        total_size = len(dataset)
        train_size = int(total_size * frac)
        val_size = total_size - train_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        # Model setup
        model = model_class(num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Train for few epochs
        model.train()
        for epoch in range(5):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        # Evaluate on train subset
        model.eval()
        with torch.no_grad():
            train_preds, train_labels = [], []
            for images, labels in train_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.numpy())
            train_acc = accuracy_score(train_labels, train_preds)
            train_accuracies.append(train_acc)
            val_preds, val_labels = [], []
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
            val_acc = accuracy_score(val_labels, val_preds)
            val_accuracies.append(val_acc)
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_fractions, train_accuracies, 'o-', label='Training Accuracy', color='cornflowerblue')
    plt.plot(train_fractions, val_accuracies, 'o-', label='Validation Accuracy', color='darkorange')
    plt.title(title)
    plt.xlabel('Training Set Fraction')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    save_plot_to_pdf(title, plt.gcf())
    plt.close()

# ------------------------
# Prediction Uncertainty Plot
# ------------------------

def plot_prediction_uncertainty_cnn(y_true, y_score, model_name):
    lower_uncertainty = 0.4
    upper_uncertainty = 0.6
    uncertainty_mask = (y_score >= lower_uncertainty) & (y_score <= upper_uncertainty)
    uncertainty_percent = np.mean(uncertainty_mask) * 100
    uncertainty_color = 'orange'
    uncertainty_label = f"Uncertainty: {uncertainty_percent:.2f}%"
    plt.figure(figsize=(12, 7))
    plt.hist(y_score, bins=100, alpha=0.5, color="slateblue", density=True, label="Max Confidence")
    sns.kdeplot(y_score, fill=False, color="darkblue", lw=2, bw_adjust=1.5)
    plt.axvline(0.5, color='black', linestyle='--')
    plt.axvspan(lower_uncertainty, upper_uncertainty, color=uncertainty_color, alpha=0.3,
                label=f"Uncertainty Area ({lower_uncertainty}-{upper_uncertainty})")
    plt.xlabel("Max Probability (Confidence)")
    plt.ylabel("Density")
    plt.title(f"Uncertainty - {model_name}")
    empty_patch = Patch(facecolor='none', edgecolor='none',
                        label=rf"$\it{{{uncertainty_label.replace('%', r'\%')}}}$")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(empty_patch)
    plt.legend(handles=handles, loc='upper left', frameon=True)
    save_plot_to_pdf(f"{model_name} Prediction Uncertainty", plt.gcf())
    plt.close()

# ------------------------
# Distribution Plot
# ------------------------

def plot_class_distribution(dataset, class_names):
    from collections import Counter
    label_counts = Counter(dataset.labels)
    counts = [label_counts[i] for i in range(len(class_names))]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=counts, hue=class_names, palette="mako", legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title("Images Distribution", fontsize=16)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.tight_layout()
    save_plot_to_pdf("Distribuzione Classi", plt.gcf())
    plt.close()

def plot_class_pie(dataset, class_names):
    from collections import Counter
    label_counts = Counter(dataset.labels)
    counts = [label_counts[i] for i in range(len(class_names))]
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(counts, labels=class_names, autopct='%1.1f%%',
                                       startangle=140, colors=sns.color_palette("tab10"),
                                       wedgeprops=dict(width=0.4), pctdistance=0.85)
    plt.setp(autotexts, size=10, weight="bold")
    plt.title("Images Distribution Percentage", fontsize=16)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.tight_layout()
    save_plot_to_pdf("Distribuzione Percentuale Classi", plt.gcf())
    plt.close()

# ------------------------
# Save Plot to PDF
# ------------------------

def analysis(all_labels, all_preds):
    print("Images distribution:")
    print(Counter(dataset.labels))
    # Dataset statistics
    plot_class_distribution(dataset, class_names)
    plot_class_pie(dataset, class_names)

    # CNN analysis
    # Classification report e Confusion matrix
    print(classification_report(all_labels, all_preds, target_names=class_names))
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Analysis/CNN Classification Report.csv')
    print("Saved: 'CNN Classification Report.csv'")
    plot_confusion_matrix(all_labels, all_preds, class_names, title="CNN Confusion Matrix")
    # Metrics
    plot_training_metrics('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Training metrics.npz', model_name="CNN Training")
    plot_training_metrics('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Fine-tune metrics.npz', model_name="CNN FineTuning")
    # Learning curve plot
    plot_learning_curve_cnn(TrashCNN, dataset, device, class_names, title="CNN Learning Curve") 
    # Uncertainty plot
    all_probs = []
    all_labels_unc = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1).values.cpu().numpy()
            all_probs.extend(max_probs)
            all_labels_unc.extend(labels.numpy()) 
    plot_prediction_uncertainty_cnn(all_labels_unc, np.array(all_probs), model_name="CNN")

    # LR-SGD analysis
    # Classification report e Confusion matrix
    lrsgd_features = []
    lrsgd_true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = extract_features_for_lrsgd(model, images)
            feats_scaled = scaler.transform(feats)
            preds = lrsgd_model.predict(feats_scaled)
            lrsgd_features.extend(preds)
            lrsgd_true_labels.extend(labels.cpu().numpy())
    lrsgd_report = classification_report(lrsgd_true_labels, lrsgd_features, target_names=class_names, output_dict=True)
    df_lrsgd = pd.DataFrame(lrsgd_report).transpose()
    df_lrsgd.to_csv('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Analysis/Lrsgd Classification Report.csv')
    print(classification_report(lrsgd_true_labels, lrsgd_features, target_names=class_names))
    print("Saved: 'Lrsgd Classification Report.csv'")
    plot_confusion_matrix(lrsgd_true_labels, lrsgd_features, class_names, title="LR-SGD Confusion Matrix")
    # Metrics
    plot_training_metrics('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/lrsgd metrics.npz', model_name="lrsgd Training")

def save_plot_to_pdf(filename, fig=None, directory = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Analysis'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_file = os.path.join(directory, f"{filename}.pdf")
    if fig:
        fig.savefig(path_file, format='pdf', bbox_inches='tight')
    else:
        plt.savefig(path_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {path_file}")

# ------------------------
# Gradio Interface
# ------------------------

def build_gradio_interface(model, class_names, device):
    transform_for_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    category_colors = {
        "battery": "#FF5252",     
        "biological": "#4CAF50",  
        "plastic": "#2196F3",     
        "glass": "#00BCD4",       
        "metal": "#9E9E9E",       
        "paper": "#FFC107",       
        "clothes": "#E91E63"      
    }

    category_icons = {
        "battery": "🔋",
        "biological": "🌿",
        "plastic": "🥤",
        "glass": "🍷",
        "metal": "🔩",
        "paper": "📄",
        "clothes": "👕"
    }

    def predict(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform_for_input(image).unsqueeze(0).to(device)

        # CNN prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            category_cnn = class_names[pred.item()]
            cnn_confidence = confidence.item()

        # LR-SGD prediction
        features = extract_features_for_lrsgd(model, image_tensor)
        features_scaled = scaler.transform(features)
        lrsgd_probs = lrsgd_model.predict_proba(features_scaled)
        lrsgd_confidence = np.max(lrsgd_probs)
        lrsgd_pred = np.argmax(lrsgd_probs)
        lrsgd_category = class_names[lrsgd_pred]

        if cnn_confidence >= lrsgd_confidence:
            final_category = category_cnn
        else:
            final_category = lrsgd_category

        color = category_colors.get(final_category.lower(), "#FFFFFF")
        icon = category_icons.get(final_category.lower(), "♻️")

        final_result = f"{icon} {final_category.upper()} {icon}"
        confidence_value = max(cnn_confidence, lrsgd_confidence)
        confidence_bar = f"▰" * int(confidence_value * 10) + f" ({confidence_value*100:.1f}%)"

        return (
            f"<span style='color: {category_colors[category_cnn.lower()]}; font-weight: bold;'>🧠 CNN: {category_cnn.upper()}</span> {cnn_confidence*100:.1f}%",
            f"<span style='color: {category_colors[lrsgd_category.lower()]}; font-weight: bold;'>🤖 LR-SGD: {lrsgd_category.upper()}</span> {lrsgd_confidence*100:.1f}%",
            f"<h2 style='color: {color}; text-align: center;'>{final_result}</h2>",
            f"<div style='background: {color}; width: {confidence_value*100}%; height: 20px; border-radius: 10px;'></div>",
            color
        )

    with gr.Blocks(css="styles.css") as demo:
        gr.Markdown("""
            <div style='text-align: center;'>
                <h1 style='color: var(--main-color); transition: color 0.5s;'>♻️ Smart Waste Classifier</h1>
                <p style='color: #666;'>Upload a waste image to classify it correctly!</p>
            </div>
        """)
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                input_img = gr.Image(type="numpy", label="📸 Upload your photo", elem_classes="custom-upload")
                
            with gr.Column(scale=3):
                with gr.Group():
                    output_text_cnn = gr.HTML(label="Risultato CNN")
                    output_text_lrsgd = gr.HTML(label="Risultato LR-SGD")
                
                final_output_text = gr.HTML(label="Risultato Finale")
                confidence_bar = gr.HTML()
                color_state = gr.State(value="#FFFFFF")

        submit_btn = gr.Button("🚀 Start recognition", variant="primary", elem_classes="pulse-button")
        
        submit_btn.click(
            fn=predict,
            inputs=input_img,
            outputs=[output_text_cnn, output_text_lrsgd, final_output_text, confidence_bar, color_state]
        ).then(
            None,
            inputs=[color_state],
            outputs=[],
            js="""
            (color) => {
                document.documentElement.style.setProperty('--main-color', color);
                document.querySelector('.pulse-button').style.backgroundColor = color;
                document.querySelector('h1').style.color = color;
                
                const btn = document.querySelector('.pulse-button');
                btn.classList.add('animate-pulse');
                setTimeout(() => btn.classList.remove('animate-pulse'), 1000);
            }
            """
        )

    return demo.launch()

# ------------------------
# Main Execution
# ------------------------

if __name__ == "__main__":
    root_dir = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/Data'
    dataset = TrashDataset(root_dir, transform=train_transform)
    
    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset.labels), y=dataset.labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = TrashCNN(num_classes=len(class_names)).to(device)

    # Load trained model weights, LR-SGD and scaler
    model.load_state_dict(torch.load('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Trash classifier finetuned.pth', map_location=device))
    model.eval()
    lrsgd_model = joblib.load('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Lrsgd model.pth')
    scaler = joblib.load('/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Utils/Scaler.pth')

    all_labels, all_preds = evaluate_model(model, test_loader, device, class_names)

    # analysis(all_labels, all_preds) # function to perform analysis and save plots

    build_gradio_interface(model, class_names, device)