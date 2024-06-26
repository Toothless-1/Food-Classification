from torchvision import datasets

# Assuming your dataset is in a directory called 'data/train'
dataset = datasets.ImageFolder('data')
num_classes = len(dataset.classes)
print(f"Number of classes: {num_classes}")
