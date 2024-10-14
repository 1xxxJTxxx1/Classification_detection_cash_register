import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import random
from sklearn.utils.class_weight import compute_class_weight


# Аугментация данных для классификации
aug_transforms_classification = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])
class CustomDataset(Dataset):
    def __init__(self, root, transforms=None, aug_transforms=None):
        self.root = root
        self.transforms = transforms
        self.aug_transforms = aug_transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels_detection = list(sorted(os.listdir(os.path.join(root, "labels_detection"))))
        self.labels_classification = list(sorted(os.listdir(os.path.join(root, "labels_classification"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_detection_path = os.path.join(self.root, "labels_detection", self.labels_detection[idx])
        label_classification_path = os.path.join(self.root, "labels_classification", self.labels_classification[idx])

        img = Image.open(img_path).convert("RGB")  # Открываем изображение как PIL и конвертируем в RGB

        with open(label_detection_path) as f:
            boxes = np.array([list(map(int, line.split())) for line in f])

        with open(label_classification_path) as f:
            label_classification = int(f.read().strip())

        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["class"] = torch.tensor(label_classification, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)  # Применяем трансформацию, если она задана

        # Применение аугментации данных
        if self.aug_transforms is not None and random.random() > 0.5:
            img = self.aug_transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


class CustomClassificationDataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.images = sorted(os.listdir(os.path.join(dataset_path, 'images')))
        self.labels = sorted(os.listdir(os.path.join(dataset_path, 'labels_classification')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, 'images', self.images[idx])
        label_path = os.path.join(self.dataset_path, 'labels_classification', self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            label = int(f.readline().strip())

        if self.transforms:
            image = self.transforms(image)

        return image, label

# Обучение модели детекции
def train_detection_model(dataset_path, num_epochs=10, batch_size=2, lr=0.005, momentum=0.9, weight_decay=0.0005):
    # Определение аугментации для обучения
    aug_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])
    #Загрузка данных
    dataset = CustomDataset(dataset_path, transforms=transforms.ToTensor(), aug_transforms=aug_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # Классы: фоновый класс и касса
    # Изменение классификатора модели, чтобы он соответствовал  числу классов
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    #  оптимизатор (SGD с заданными параметрами)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Обучение модели
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()   #Вычисление градиентов
            optimizer.step()    #Обновление параметров модели

            running_loss += losses.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

    torch.save(model.state_dict(), 'faster_rcnn_cassette.pth')


# Обучение модели классификации
def train_classification_model(dataset_path, num_epochs=15, batch_size=10, lr=0.00015, weight_decay=0.0001):
    # Трансформации данных
    transform_classification = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Загрузка данных
    dataset = CustomClassificationDataset(dataset_path, transforms=transform_classification)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Загрузка модели
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Считывание веса классов
    class_labels = []
    for label_file in sorted(os.listdir(os.path.join(dataset_path, 'labels_classification'))):
        with open(os.path.join(dataset_path, 'labels_classification', label_file), 'r') as f:
            class_labels.append(int(f.readline().strip()))

    unique_classes = np.unique(class_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=class_labels)

    # Преобразование class_weights в тензор PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    #  взвешенная функцию потерь
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



    # Обучение модели
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Валидация модели
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}%')

    torch.save(model.state_dict(), 'cash_register_model.pth')

# Загрузка обученной модели детекции
def load_detection_model(model_path='faster_rcnn_cassette.pth'):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Загрузка обученной модели классификации
def load_classification_model(model_path='cash_register_model.pth'):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Функция детекции и классификации состояния кассы
def detect_and_classify(image_path, detection_model, classification_model):
    # Трансформации для классификации
    transform_classification = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Открытие  и конвертация  изображение
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_np = np.array(image)

    # Масштабирование изображения для модели детекции
    transform_detection = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image_tensor = transform_detection(image).unsqueeze(0)

    # Детекция
    with torch.no_grad():
        detection_output = detection_model(image_tensor)

    boxes = detection_output[0]['boxes']
    scores = detection_output[0]['scores']

    # Порог для детекции
    threshold = 0.4
    boxes = boxes[scores > threshold].cpu().numpy()
    scores = scores[scores > threshold].cpu().numpy()

    if len(boxes) > 0:
        # bounding box с наибольшей вероятностью
        best_box_idx = np.argmax(scores)
        x1, y1, x2, y2 = map(int, boxes[best_box_idx])

        # Масштабирование координаты bounding box обратно к оригинальному размеру изображения
        x1 = int(x1 * original_size[0] / 512)
        y1 = int(y1 * original_size[1] / 512)
        x2 = int(x2 * original_size[0] / 512)
        y2 = int(y2 * original_size[1] / 512)

        cash_register = image_np[y1:y2, x1:x2]
        cash_register_pil = Image.fromarray(cash_register)
        input_tensor = transform_classification(cash_register_pil).unsqueeze(0)

        # Классификация
        with torch.no_grad():
            output = classification_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

        # Выбор класса
        if probabilities.shape[1] == 1:
            predicted_class = torch.round(probabilities).int().item()
        else:
            # Если вероятности имеют два элемента, нужно принять решение, какой класс выбрать
            #  Используем класс с наибольшей вероятностью
            predicted_class = torch.argmax(probabilities).item()

        # Отладочные выводы
        if predicted_class == 1:
            return 'открыта'
        else:
            return 'закрыта'
    else:
            return 'Касса не найдена'


if __name__ == "__main__":

    dataset_path = 'dataset'
    # Обучение моделей (при необходимости)
    #train_detection_model(dataset_path, num_epochs=10)
    train_classification_model(dataset_path, num_epochs=15)

    # Загрузка моделей
    detection_model = load_detection_model('faster_rcnn_cassette.pth')
    classification_model = load_classification_model('cash_register_model.pth')

    # Детекция и классификация состояния кассы
    result = detect_and_classify('D:/test1.png', detection_model, classification_model)
    print(result)