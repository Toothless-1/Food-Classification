import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import datasets, models, transforms
from PIL import Image
import torch
import io
import os

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка предобученной модели MobileNet
#model = models.mobilenet_v2(pretrained=True)
#model.load_state_dict(torch.load('mobilenet_v2_food101.pt'))

# Изменение последнего слоя модели для 4 классов (пример)
#num_classes = 32  # Укажите количество ваших классов
#model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
#model.eval()

# Определение количества классов в вашем датасете
#data_dir = '/app/data/'  # Путь к примонтированному каталогу
#train_dataset = datasets.ImageFolder(os.path.join(data_dir, ''))
#num_classes = len(train_dataset.classes)
#model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
#model.load_state_dict(torch.load('mobilenet_v2_food101.pt'))
#model.eval()

# Load the pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)  # 101 classes for Food-101 dataset

# Load the trained model weights
model_path = 'mobilenet_v2_food101.pt'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


# Трансформации данных
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Классы (для примера, замените на свои классы)
# Определение имен классов
#class_names = train_dataset.classes
class_names = ["apple_pie",
"baby_back_ribs",
"baklava",
]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/pytorch")
def pytorch_info():
    return {
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Devices": torch.cuda.device_count()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename} with content type: {file.content_type}")

    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.error("Invalid file type")
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        logger.info(f"Image size: {image.size}")
        image = data_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_class_idx = preds[0].item()
            predicted_class_name = class_names[predicted_class_idx]
            logger.info(f"Predicted class index: {predicted_class_idx}, class name: {predicted_class_name}")

        return JSONResponse(content={"predicted_class": predicted_class_name})

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)