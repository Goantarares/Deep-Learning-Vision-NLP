import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score

print("=" * 70)
print("VERIFICARE MEDIU KAGGLE")
print("=" * 70)

if os.path.exists('/kaggle/input'):
    BASE_PATH = Path('/kaggle/input')
    OUTPUT_DIR = '/kaggle/working/raport_analiza'
    print("Mediu Kaggle")
    print(f"Input path: {BASE_PATH}")
    print(f"Output path: {OUTPUT_DIR}")

    # Listare datasets disponibile
    print("\nDatasets disponibile:")
    for item in BASE_PATH.iterdir():
        print(f"  - {item.name}")
else:
    BASE_PATH = Path('.')
    OUTPUT_DIR = 'raport_analiza'
    print("Mediu local detectat")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONFIGURARI

IMAGEBITS_CLASSES = {
    '1': 'airplane', '2': 'bird', '3': 'car', '4': 'cat', '5': 'deer',
    '6': 'dog', '7': 'horse', '8': 'monkey', '9': 'ship', '10': 'truck'
}

IMAGEBITS_CLASS_NAMES = ['airplane', 'bird', 'car', 'cat', 'deer',
                         'dog', 'horse', 'monkey', 'ship', 'truck']

LANDPATCHES_CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation',
                           'Highway', 'Industrial', 'Pasture',
                           'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Verificare GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'=' * 70}")
print(f"CONFIGURARE HARDWARE")
print(f"{'=' * 70}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("ATENȚIE: GPU nu este disponibil!")

# Hyperparametri
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
PATIENCE = 6

COMMON_IMAGE_SIZE = 96

def find_dataset(base_path, dataset_name):
    possible_names = [
        dataset_name.lower(),
        dataset_name.lower().replace('_', '-'),
        dataset_name.lower().replace('-', '_'),
        dataset_name.lower().replace(' ', '-'),
    ]

    for name in possible_names:
        for item in base_path.iterdir():
            if name in item.name.lower():
                return item

    print(f"EROARE: Nu exista dataset-ul '{dataset_name}' în {base_path}")
    print(f"Datasets disponibile:")
    for item in base_path.iterdir():
        print(f"  - {item.name}")
    return None



# FUNCȚIA DE ANALIZA

def analyze_dataset(root_path, dataset_name, img_size):
    if not os.path.exists(root_path):
        print(f"Calea {root_path} nu există.")
        return

    print(f"Explorare date pentru {dataset_name}")

    dataset = datasets.ImageFolder(
        root=root_path,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    )

    raw_classes = dataset.classes
    display_classes = [
        IMAGEBITS_CLASSES.get(c, c) for c in raw_classes
    ] if "Imagebits" in dataset_name else raw_classes

    # Echilibrul claselor
    class_counts = [0] * len(raw_classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    plt.figure(figsize=(12, 5))
    plt.bar(display_classes, class_counts, color='teal', edgecolor='black')
    plt.title(f"Echilibrul Claselor - {dataset_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Clase", fontsize=12)
    plt.ylabel("Numar de imagini", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/{dataset_name.replace(' ', '_')}_echilibru.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nDistributia claselor:")
    total_images = sum(class_counts)
    for i, cls in enumerate(display_classes):
        percentage = 100 * class_counts[i] / total_images
        print(f"  {cls:25s}: {class_counts[i]:5d} imagini ({percentage:5.2f}%)")

    # Analiza pixeli
    print(f"\nAnaliza statistica a pixelilor (sample: min(1000, {len(dataset)})):")
    means = [[] for _ in range(3)]
    stds = [[] for _ in range(3)]

    sample_size = min(1000, len(dataset))
    for i in range(sample_size):
        img, _ = dataset[i]
        for c in range(3):
            means[c].append(img[c].mean().item())
            stds[c].append(img[c].std().item())

    for c, color in enumerate(['R', 'G', 'B']):
        print(f"  Canal {color}: mean={np.mean(means[c]):.3f}, std={np.mean(stds[c]):.3f}")

    # Grila variabilitate (5 exemple per clasa)
    n_rows, n_cols = len(raw_classes), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows):
        indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label == i]
        n_samples = min(n_cols, len(indices))
        selected = np.random.choice(indices, n_samples, replace=False)

        for j in range(n_cols):
            if j < n_samples:
                img, _ = dataset[selected[j]]
                img = img.permute(1, 2, 0).numpy()
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].text(-15, img_size // 2, display_classes[i],
                                    va='center', ha='right', fontweight='bold', fontsize=10)
            else:
                axes[i, j].axis('off')

    plt.suptitle(f"Variabilitate intra-clasa - {dataset_name}", fontsize=16, fontweight='bold')
    save_path = f"{OUTPUT_DIR}/{dataset_name.replace(' ', '_')}_variabilitate.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nAnaliza finalizata pentru {dataset_name}")
    print(f"Grafice salvate in '{OUTPUT_DIR}/'")


class AlbumentationsDataset(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.alb_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.alb_transform:
            augmented = self.alb_transform(image=image)
            image = augmented['image']
        return image, target



# MLP

class MLP_Model(nn.Module):

    def __init__(self, input_dim, num_classes=10):
        super(MLP_Model, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ARHITECTURA CNN (LeNet Modern)

class LeNet_Modern(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet_Modern, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# FUNCȚIE DE ANTRENARE

def run_training(model, train_loader, test_loader, desc, epochs=EPOCHS, lr=LR, save_path=None, val_loader=None,
                 patience=PATIENCE):

    print(f"Antrenare: {desc}")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    acc_history = []
    loss_history = []
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    eval_loader = val_loader if val_loader is not None else test_loader
    eval_name = "Val" if val_loader is not None else "Test"

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        loss_history.append(avg_train_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in eval_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        eval_acc = 100 * correct / total
        acc_history.append(eval_acc)
        scheduler.step(eval_acc)

        # Early stopping
        if eval_acc > best_val_acc:
            best_val_acc = eval_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improvement = "^"
        else:
            patience_counter += 1
            improvement = ""

        lr_indicator = f" [LR={current_lr:.6f}]" if current_lr != lr else ""
        print(f"Epoca {epoch + 1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"{eval_name} Acc={eval_acc:.2f}% {improvement}{lr_indicator}")

        if patience_counter >= patience:
            print(f"\nEarly Stopping!")
            print(f"   Best {eval_name} Acc: {best_val_acc:.2f}% (epoca {epoch + 1 - patience})")
            break

    if save_path:
        save_path = os.path.join('/kaggle/working', save_path)
        if best_model_state is not None:
            torch.save(best_model_state, save_path)
        print(f"✓ Model salvat: {save_path}")

    final_test_acc = None
    if val_loader is not None:
        print(f"EVALUARE FINALA PE TEST SET")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        final_test_acc = 100 * correct / total
        print(f"Acuratețe finală pe TEST: {final_test_acc:.2f}%")
        print(f"(Best Val Acc era: {best_val_acc:.2f}%)")

    return acc_history, loss_history, final_test_acc, best_model_state



# 8. FUNCȚIE EVALUARE

def evaluate_model(model, test_loader):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    return accuracy, f1, all_preds, all_labels



# FUNCȚIE CONFUSION MATRIX

def generate_confusion_matrix(all_labels, all_preds, class_names, title, save_name):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Număr predicții'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Adevărat', fontsize=14)
    plt.xlabel('Prezis', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


# ==========================================
# 10. MAIN
# ==========================================
if __name__ == "__main__":


    imagebits_path = None
    land_patches_path = None

    if BASE_PATH.name == 'input':
        imagebits_dataset = find_dataset(BASE_PATH, 'imagebits')
        land_patches_dataset = find_dataset(BASE_PATH, 'land-patches')

        if imagebits_dataset:
            imagebits_path = imagebits_dataset / 'imagebits'
            if not imagebits_path.exists():
                imagebits_path = imagebits_dataset

        if land_patches_dataset:
            land_patches_path = land_patches_dataset / 'land_patches'
            if not land_patches_path.exists():
                land_patches_path = land_patches_dataset
    else:
        imagebits_path = Path('./imagebits')
        land_patches_path = Path('./land_patches')

    if imagebits_path and imagebits_path.exists():
        print(f"\nStructura Imagebits:")
        print(f"  Train: {(imagebits_path / 'train').exists()}")
        print(f"  Test: {(imagebits_path / 'test').exists()}")
    else:
        print("EROARE: Imagebits nu a fost gasit!")
        sys.exit(1)

    if land_patches_path and land_patches_path.exists():
        print(f"\nStructură Land Patches:")
        print(f"  Train: {(land_patches_path / 'train').exists()}")
        print(f"  Val: {(land_patches_path / 'val').exists()}")
        print(f"  Test: {(land_patches_path / 'test').exists()}")
    else:
        print("EROARE: Land Patches nu a fost gasit!")
        sys.exit(1)

    # EXPLORAREA DATELOR

    print(f"\n{'=' * 70}")
    print("FAZA 1: EXPLORAREA DATELOR")
    print(f"{'=' * 70}")

    analyze_dataset(str(imagebits_path / 'train'), "Imagebits_Train", COMMON_IMAGE_SIZE)
    analyze_dataset(str(land_patches_path / 'train'), "Land_Patches_Train", COMMON_IMAGE_SIZE)
    analyze_dataset(str(land_patches_path / 'val'), "Land_Patches_Val", COMMON_IMAGE_SIZE)
    analyze_dataset(str(land_patches_path / 'test'), "Land_Patches_Test", COMMON_IMAGE_SIZE)


    trans_no_aug = A.Compose([
        A.Resize(COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    trans_mlp_ib = A.Compose([
        A.Resize(COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomResizedCrop(size=(COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    trans_cnn_ib = A.Compose([
        A.Resize(COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomResizedCrop(size=(COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    trans_lp = A.Compose([
        A.Resize(COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    results = []

    # MLP - ANTRENARE

    print(f"\n{'=' * 70}")
    print("FAZA 2: ANTRENARE MLP")
    print(f"{'=' * 70}")

    # MLP Imagebits - fara augmentari
    print("\n[2.1] MLP Imagebits - Baseline (fără augmentări)")
    loader_train_mlp_ib_no_aug = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'train'), trans_no_aug),
        BATCH_SIZE, shuffle=True, num_workers=2
    )
    loader_test_mlp_ib_no_aug = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'test'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )

    model_mlp_no_aug = MLP_Model(input_dim=3 * COMMON_IMAGE_SIZE * COMMON_IMAGE_SIZE)
    history_mlp_ib_no_aug, loss_mlp_ib_no_aug, _, _ = run_training(
        model_mlp_no_aug, loader_train_mlp_ib_no_aug, loader_test_mlp_ib_no_aug,
        "MLP Imagebits - FĂRĂ augmentări (baseline)"
    )

    acc_mlp_no_aug, f1_mlp_no_aug, _, _ = evaluate_model(model_mlp_no_aug, loader_test_mlp_ib_no_aug)
    results.append({
        'Model': 'MLP',
        'Dataset': 'Imagebits',
        'Configurație': 'Baseline (fără augmentări)',
        'Test Accuracy (%)': f'{acc_mlp_no_aug:.2f}',
        'Test F1-Score (%)': f'{f1_mlp_no_aug:.2f}',
        'Augmentări': 'Nu',
        'Transfer Learning': 'Nu'
    })

    # MLP Imagebits - cu augmentari
    print("\nMLP Imagebits - Cu augmentări")
    loader_train_mlp_ib = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'train'), trans_mlp_ib),
        BATCH_SIZE, shuffle=True, num_workers=2
    )
    loader_test_mlp_ib = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'test'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )

    model_mlp_ib = MLP_Model(input_dim=3 * COMMON_IMAGE_SIZE * COMMON_IMAGE_SIZE)
    history_mlp_ib, loss_mlp_ib, _, _ = run_training(
        model_mlp_ib, loader_train_mlp_ib, loader_test_mlp_ib,
        "MLP Imagebits - CU augmentări",
        save_path="mlp_imagebits_pretrained.pth"
    )

    acc_mlp_ib, f1_mlp_ib, preds_mlp_ib, labels_mlp_ib = evaluate_model(model_mlp_ib, loader_test_mlp_ib)
    results.append({
        'Model': 'MLP',
        'Dataset': 'Imagebits',
        'Configurație': 'Cu augmentări (HFlip, RBC, RRC, CJ)',
        'Test Accuracy (%)': f'{acc_mlp_ib:.2f}',
        'Test F1-Score (%)': f'{f1_mlp_ib:.2f}',
        'Augmentări': 'Da',
        'Transfer Learning': 'Nu'
    })

    # Grafic comparativ MLP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(history_mlp_ib_no_aug, label='Fără augmentări', marker='o', linewidth=2)
    ax1.plot(history_mlp_ib, label='Cu augmentări', marker='s', linewidth=2)
    ax1.set_title('MLP Imagebits: Impact Augmentări - Acuratețe', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoca', fontsize=12)
    ax1.set_ylabel('Acuratețe (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_mlp_ib_no_aug, label='Fără augmentări', marker='o', linewidth=2)
    ax2.plot(loss_mlp_ib, label='Cu augmentări', marker='s', linewidth=2)
    ax2.set_title('MLP Imagebits: Impact Augmentări - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoca', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mlp_augmentation_comparison.png", dpi=150)
    plt.close()

    # MLP LandPatches - Fine-tuned
    print("\nMLP Fine-tuning pe Land Patches")
    loader_train_mlp_lp = DataLoader(
        AlbumentationsDataset(str(land_patches_path / 'train'), trans_lp),
        BATCH_SIZE, shuffle=True, num_workers=2
    )
    loader_val_mlp_lp = DataLoader(
        AlbumentationsDataset(str(land_patches_path / 'val'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )
    loader_test_mlp_lp = DataLoader(
        AlbumentationsDataset(str(land_patches_path / 'test'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )

    model_mlp_lp = MLP_Model(input_dim=3 * COMMON_IMAGE_SIZE * COMMON_IMAGE_SIZE)
    model_mlp_lp.load_state_dict(torch.load("/kaggle/working/mlp_imagebits_pretrained.pth"))
    history_mlp_lp, loss_mlp_lp, final_test_acc_mlp, state_mlp_lp = run_training(
        model_mlp_lp, loader_train_mlp_lp, loader_test_mlp_lp,
        "MLP Land Patches - Fine-tuned",
        lr=LR / 10,
        val_loader=loader_val_mlp_lp,
        save_path="mlp_landpatches_finetuned.pth"
    )

    # Reload best model
    model_mlp_lp.load_state_dict(state_mlp_lp)
    _, f1_mlp_lp, preds_mlp_lp, labels_mlp_lp = evaluate_model(model_mlp_lp, loader_test_mlp_lp)

    results.append({
        'Model': 'MLP',
        'Dataset': 'Land Patches',
        'Configurație': 'Fine-tuned (HFlip, VFlip, Rot90, Blur, Gamma)',
        'Test Accuracy (%)': f'{final_test_acc_mlp:.2f}',
        'Test F1-Score (%)': f'{f1_mlp_lp:.2f}',
        'Augmentări': 'Da',
        'Transfer Learning': 'Da (Imagebits)'
    })

    # Grafic fine-tuning MLP
    plt.figure(figsize=(12, 6))
    plt.plot(history_mlp_ib, label='Imagebits (source)', marker='o', linewidth=2)
    plt.plot(history_mlp_lp, label='Land Patches (fine-tuned)', marker='s', linewidth=2)
    plt.title('MLP: Transfer Learning Imagebits → Land Patches', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Acuratețe (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mlp_fine_tuning_results.png", dpi=150)
    plt.close()

    # CNN - ANTRENARE

    print("FAZA 3: ANTRENARE CNN")

    # 3.1 CNN Imagebits - FĂRĂ augmentări
    print("\n[3.1] CNN Imagebits - Baseline (fără augmentări)")
    loader_train_cnn_ib_no_aug = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'train'), trans_no_aug),
        BATCH_SIZE, shuffle=True, num_workers=2
    )
    loader_test_cnn_ib_no_aug = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'test'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )

    model_cnn_no_aug = LeNet_Modern(num_classes=10)
    history_cnn_ib_no_aug, loss_cnn_ib_no_aug, _, _ = run_training(
        model_cnn_no_aug, loader_train_cnn_ib_no_aug, loader_test_cnn_ib_no_aug,
        "CNN Imagebits - FĂRĂ augmentări (baseline)"
    )

    acc_cnn_no_aug, f1_cnn_no_aug, _, _ = evaluate_model(model_cnn_no_aug, loader_test_cnn_ib_no_aug)
    results.append({
        'Model': 'CNN',
        'Dataset': 'Imagebits',
        'Configurație': 'Baseline (fără augmentări)',
        'Test Accuracy (%)': f'{acc_cnn_no_aug:.2f}',
        'Test F1-Score (%)': f'{f1_cnn_no_aug:.2f}',
        'Augmentări': 'Nu',
        'Transfer Learning': 'Nu'
    })

    # CNN Imagebits
    print("\n[3.2] CNN Imagebits - Cu augmentări")
    loader_train_cnn_ib = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'train'), trans_cnn_ib),
        BATCH_SIZE, shuffle=True, num_workers=2
    )
    loader_test_cnn_ib = DataLoader(
        AlbumentationsDataset(str(imagebits_path / 'test'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )

    model_cnn_ib = LeNet_Modern(num_classes=10)
    history_cnn_ib, loss_cnn_ib, _, _ = run_training(
        model_cnn_ib, loader_train_cnn_ib, loader_test_cnn_ib,
        "CNN Imagebits - CU augmentări",
        save_path="cnn_imagebits_pretrained.pth"
    )

    acc_cnn_ib, f1_cnn_ib, preds_cnn_ib, labels_cnn_ib = evaluate_model(model_cnn_ib, loader_test_cnn_ib)
    results.append({
        'Model': 'CNN',
        'Dataset': 'Imagebits',
        'Configurație': 'Cu augmentări (HFlip, RBC, SSR, RRC, CJ)',
        'Test Accuracy (%)': f'{acc_cnn_ib:.2f}',
        'Test F1-Score (%)': f'{f1_cnn_ib:.2f}',
        'Augmentări': 'Da',
        'Transfer Learning': 'Nu'
    })

    # Grafic comparativ CNN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(history_cnn_ib_no_aug, label='Fără augmentări', marker='o', linewidth=2)
    ax1.plot(history_cnn_ib, label='Cu augmentări', marker='s', linewidth=2)
    ax1.set_title('CNN Imagebits: Impact Augmentări - Acuratețe', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoca', fontsize=12)
    ax1.set_ylabel('Acuratețe (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_cnn_ib_no_aug, label='Fără augmentări', marker='o', linewidth=2)
    ax2.plot(loss_cnn_ib, label='Cu augmentări', marker='s', linewidth=2)
    ax2.set_title('CNN Imagebits: Impact Augmentări - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoca', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cnn_augmentation_comparison.png", dpi=150)
    plt.close()

    # 3.3 CNN Land Patches - Fine-tuned
    print("\n[3.3] CNN Fine-tuning pe Land Patches")
    loader_train_cnn_lp = DataLoader(
        AlbumentationsDataset(str(land_patches_path / 'train'), trans_lp),
        BATCH_SIZE, shuffle=True, num_workers=2
    )
    loader_val_cnn_lp = DataLoader(
        AlbumentationsDataset(str(land_patches_path / 'val'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )
    loader_test_cnn_lp = DataLoader(
        AlbumentationsDataset(str(land_patches_path / 'test'), trans_no_aug),
        BATCH_SIZE, num_workers=2
    )

    model_cnn_lp = LeNet_Modern(num_classes=10)
    model_cnn_lp.load_state_dict(torch.load("/kaggle/working/cnn_imagebits_pretrained.pth"))
    history_cnn_lp, loss_cnn_lp, final_test_acc_cnn, state_cnn_lp = run_training(
        model_cnn_lp, loader_train_cnn_lp, loader_test_cnn_lp,
        "CNN Land Patches - Fine-tuned",
        lr=LR / 10,
        val_loader=loader_val_cnn_lp,
        save_path="cnn_landpatches_finetuned.pth"
    )

    # Reload best model
    model_cnn_lp.load_state_dict(state_cnn_lp)
    _, f1_cnn_lp, preds_cnn_lp, labels_cnn_lp = evaluate_model(model_cnn_lp, loader_test_cnn_lp)

    results.append({
        'Model': 'CNN',
        'Dataset': 'Land Patches',
        'Configurație': 'Fine-tuned (HFlip, VFlip, Rot90, RBC, Blur, Gamma)',
        'Test Accuracy (%)': f'{final_test_acc_cnn:.2f}',
        'Test F1-Score (%)': f'{f1_cnn_lp:.2f}',
        'Augmentări': 'Da',
        'Transfer Learning': 'Da (Imagebits)'
    })

    # Grafic fine-tuning CNN
    plt.figure(figsize=(12, 6))
    plt.plot(history_cnn_ib, label='Imagebits (source)', marker='o', linewidth=2)
    plt.plot(history_cnn_lp, label='Land Patches (fine-tuned)', marker='s', linewidth=2)
    plt.title('CNN: Transfer Learning Imagebits → Land Patches', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Acuratețe (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cnn_fine_tuning_results.png", dpi=150)
    plt.close()

    print("TABEL REZULTATE FINALE")

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    df_results.to_csv(f"{OUTPUT_DIR}/tabel_rezultate.csv", index=False)

    print("GENERARE CONFUSION MATRICES")

    # MLP Imagebits (cel mai bun = cu augmentări)
    generate_confusion_matrix(
        labels_mlp_ib, preds_mlp_ib, IMAGEBITS_CLASS_NAMES,
        'Confusion Matrix - MLP Imagebits (cu augmentări)',
        'cm_mlp_imagebits'
    )

    # CNN Imagebits (cel mai bun = cu augmentări)
    generate_confusion_matrix(
        labels_cnn_ib, preds_cnn_ib, IMAGEBITS_CLASS_NAMES,
        'Confusion Matrix - CNN Imagebits (cu augmentări)',
        'cm_cnn_imagebits'
    )

    # MLP Land Patches (fine-tuned pe TEST)
    generate_confusion_matrix(
        labels_mlp_lp, preds_mlp_lp, LANDPATCHES_CLASS_NAMES,
        'Confusion Matrix - MLP Land Patches Fine-tuned (TEST)',
        'cm_mlp_landpatches'
    )

    # CNN Land Patches (fine-tuned pe TEST)
    generate_confusion_matrix(
        labels_cnn_lp, preds_cnn_lp, LANDPATCHES_CLASS_NAMES,
        'Confusion Matrix - CNN Land Patches Fine-tuned (TEST)',
        'cm_cnn_landpatches'
    )
