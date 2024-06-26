import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

# Путь к файлам с списками изображений и меток
train_txt_path = 'meta/train3.txt'
test_txt_path = 'meta/test3.txt'
data_root = 'data'

# Чтение списка файлов и меток для обучающего набора данных
with open(train_txt_path, 'r') as f:
    train_lines = f.readlines()

# Преобразование данных в список кортежей (путь к файлу, метка) для обучающего набора данных
train_data = [(os.path.join(data_root, line.strip() + '.jpg'), line.strip().split('/')[0]) for line in train_lines]

# Чтение списка файлов и меток для валидационного набора данных
with open(test_txt_path, 'r') as f:
    test_lines = f.readlines()

# Преобразование данных в список кортежей (путь к файлу, метка) для валидационного набора данных
test_data = [(os.path.join(data_root, line.strip() + '.jpg'), line.strip().split('/')[0]) for line in test_lines]

# Создание словаря меток
labels = sorted(set([label for _, label in train_data]))
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Определение трансформаций данных
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Определение пользовательского набора данных
class FoodDataset(Dataset):
    def __init__(self, data, label_to_idx, transform=None):
        self.data = data
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx

# Создание обучающего и валидационного наборов данных
train_dataset = FoodDataset(train_data, label_to_idx, transform=data_transforms)
test_dataset = FoodDataset(test_data, label_to_idx, transform=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print("Classes:", labels)
# Определение модели, функции потерь и оптимизатора
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 10 == 0:  # вывод каждые 10 батчей
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch + 1}/{num_epochs}] completed, Average Training Loss: {running_loss / len(train_dataloader):.4f}')

    # Валидация модели
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}] completed, Validation Loss: {val_loss / len(test_dataloader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Сохранение модели
model_save_path = 'mobilenet_v2_food101.pt'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
