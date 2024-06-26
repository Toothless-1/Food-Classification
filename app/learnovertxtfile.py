import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

# Путь к файлу с списком изображений и меток
#train_txt_path = '/home/dreamer/food-101/meta/train.txt'
#train_txt_path = 'meta/train.txt'
train_txt_path = 'meta/train2.txt'
#data_root = '/home/dreamer/food-101/images'
data_root = 'data'

# Чтение списка файлов и меток
with open(train_txt_path, 'r') as f:
    lines = f.readlines()

# Преобразование данных в список кортежей (путь к файлу, метка)
data = [(os.path.join(data_root, line.strip() + '.jpg'), line.strip().split('/')[0]) for line in lines]

# Создание словаря меток
labels = sorted(set([label for _, label in data]))
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

# Создание набора данных и загрузчика данных
dataset = FoodDataset(data, label_to_idx, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Определение модели, функции потерь и оптимизатора
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 10 == 0:  # вывод каждые 10 батчей
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch + 1}/{num_epochs}] completed, Average Loss: {running_loss / len(dataloader):.4f}')

# Сохранение модели
model_save_path = 'mobilenet_v2_food101.pt'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
