import re

def process_text(file_path):
    # Открываем файл для чтения
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Преобразуем текст в нижний регистр
    text = text.lower()
    
    # Удаляем все знаки кроме букв, цифр, пробелов, точек с запятой и новой строки
    text = re.sub(r'[^a-zа-яё0-9 ;\n]', '', text)
    
    # Сохраняем результат в новый файл
    with open('processed_dataset.txt', 'w', encoding='utf-8') as file:
        file.write(text)

# Укажите путь к вашему файлу
file_path = 'dataset.txt'
process_text(file_path)