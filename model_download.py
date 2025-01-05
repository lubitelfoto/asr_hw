import os

import requests


def download_file():
    url = "https://drive.google.com/file/d/1ogXHNYqTyoa-xz8Z6TA4CR0KO6pWILJc/view?usp=sharing"

    save_dir = "/new_saved/testing_1/"
    file_name = "model_best.pth"

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, file_name)

    try:
        print(f"Скачивание файла с {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверяем, что запрос прошел успешно

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Файл сохранен в: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при скачивании файла: {e}")


if __name__ == "__main__":
    download_file()
