import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Определение простой модели (тот же класс SimpleNN)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_labels)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Загружаем данные, чтобы использовать тот же vectorizer
data_path = 'dataset.txt'
data = pd.read_csv(data_path, sep=';')
text_data = data['текст']
labels = data['класс0']

# Векторизация текста
vectorizer = CountVectorizer(max_features=10000)
vectorizer.fit(text_data)

# Проверка количества классов
num_labels = labels.nunique()

# Определение и загрузка модели
input_dim = len(vectorizer.get_feature_names_out())
model = SimpleNN(input_dim, num_labels)
model.load_state_dict(torch.load(f"class_first123123.pth"))
model.eval()  # Устанавливаем режим оценки

# Устройство (CUDA или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    # Векторизация входного текста
    X = vectorizer.transform([text]).toarray()
    input_ids = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# Прием текста из консоли и предсказание класса
import sys

if __name__ == "__main__":
    print("Введите текст для классификации:")
    input_text = input()
    predicted_class = predict(input_text)
    print(f"Предсказанный класс: {predicted_class}")