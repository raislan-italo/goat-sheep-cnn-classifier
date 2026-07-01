import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# Hiperparâmetros
SEED = 42
BATCH_SIZE = 16                
EPOCHS = 30                  
LR = 1e-3                  
IMG_SIZE = 128           
PATIENCE = 6 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
GENERATOR = torch.Generator().manual_seed(SEED)

# Transformações com aumento de dados
def get_train_transforms():

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# Transformações sem uumento de dados
def get_eval_transforms():

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# Permite aplicar transforms diferentes a cada Subset gerado pelo random_split
class TransformedSubset(torch.utils.data.Dataset):

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.subset.dataset.samples[self.subset.indices[index]]
        img = self.subset.dataset.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

# Dataset
def load_dataset(path):
    base_dataset = datasets.ImageFolder(root=path)
    class_names = base_dataset.classes

    size_dataset = len(base_dataset)

    # Tamanho das divisões: treino(70%), validação(15%), teste(15%)
    train_size = int(0.7 * size_dataset)
    val_size = int(0.15 * size_dataset)
    test_size = size_dataset - train_size - val_size

    train_sub, val_sub, test_sub = random_split(
        base_dataset, [train_size, val_size, test_size], generator=GENERATOR
    )

    train_ds = TransformedSubset(train_sub, get_train_transforms())
    val_ds = TransformedSubset(val_sub, get_eval_transforms())
    test_ds = TransformedSubset(test_sub, get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,num_workers=2, pin_memory=True)

    print(f"Classes: {class_names}")
    print(f"Treino: {len(train_ds)} | Validação: {len(val_ds)} | Teste: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names


def show_images(loader, class_names):
    images, labels = next(iter(loader))
    n = min(8, len(images))
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))

    for i, a in enumerate(ax.flat):
        if i >= n:
            a.axis("off")
            continue
        img = images[i].permute(1, 2, 0)
        img = (img * 0.5 + 0.5).clamp(0, 1)  # desnormaliza e garante range válido
        a.imshow(img)
        a.set_title(class_names[labels[i]])
        a.axis("off")
    plt.tight_layout()
    plt.show()

# Modelo CNN
class CNN(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # AdaptiveAvgPool2d fixa a saída em 4x4 independente do tamanho de entrada
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)  

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Treinamento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # Adiciona barra de carregamento durante o treinamento
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in bar:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = torch.sigmoid(output) > 0.5
            train_correct += (pred == labels.bool()).sum().item()
            train_total += labels.size(0)

            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = train_correct / train_total
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1} | Train Loss: {history['train_loss'][-1]:.4f} "
              f"| Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/cnn_goat_sheep.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping na época {epoch + 1} "
                      f"(sem melhora há {PATIENCE} épocas).")
                break

    # Restaura o melhor modelo
    model.load_state_dict(torch.load("models/cnn_goat_sheep.pth", map_location=DEVICE))
    return history

def validate(model, loader, criterion):
    model.eval()
    loss_total, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            output = model(images)
            loss = criterion(output, labels)
            loss_total += loss.item()
            pred = torch.sigmoid(output) > 0.5
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)

    return loss_total / len(loader), correct / total

def test_model(model, loader):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            output = model(images)
            prob = torch.sigmoid(output)
            y_score.extend(prob.cpu().numpy().flatten())
            y_pred.extend((prob > 0.5).cpu().numpy().flatten())
            y_true.extend(labels.numpy())

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
    }

    return metrics, y_true, y_pred, y_score

# Métricas separadas por classe (Goat e Sheep)
def test_class_metrics(y_true, y_pred, class_names):

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Mantém só as linhas por classe
    df = pd.DataFrame(report).transpose()
    df_classes = df.loc[class_names, ["precision", "recall", "f1-score", "support"]]
    df_classes = df_classes.round(4)

    return df_classes

# Gráficos
def plot_history(history):

    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Acurácia
    axes[0].plot(epochs_range, history["train_acc"], label="Treino", marker="o")
    axes[0].plot(epochs_range, history["val_acc"], label="Validação", marker="o")
    axes[0].set_title("Acurácia por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs_range, history["train_loss"], label="Treino", marker="o")
    axes[1].plot(epochs_range, history["val_loss"], label="Validação", marker="o")
    axes[1].set_title("Loss por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=150)
    plt.show()

def plot_confusion(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()

# Main
if __name__ == "__main__":

    train_loader, val_loader, test_loader, class_names = load_dataset("dataset/")
    show_images(train_loader, class_names)

    model = CNN(num_classes=1).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    plot_history(history)

    metrics, y_true, y_pred, y_score = test_model(model, test_loader)
    df_metrics = pd.DataFrame([metrics])

    print("\nResultado geral no teste:")
    print(df_metrics.to_string(index=False))

    df_per_class = test_class_metrics(y_true, y_pred, class_names)
    print(f"\nResultado por classe ({class_names[0]} x {class_names[1]}):")
    print(df_per_class.to_string())

    plot_confusion(y_true, y_pred, class_names)

    print("Modelo salvo em models/cnn_goat_sheep.pth"
    )