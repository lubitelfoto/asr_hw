import os

import requests


def download_file_from_google_drive(file_id, destination):
    base_url = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(base_url, params={"id": file_id}, stream=True)
    response.raise_for_status()

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(
                base_url, params={"id": file_id, "confirm": value}, stream=True
            )

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    print(f"Файл успешно сохранен в: {destination}")


def download_file():
    # https://drive.google.com/file/d/1bOAs5OpciCVf7dXNrO6qF7UcM7pVKVPI/view?usp=sharing

    file_id = "1bOAs5OpciCVf7dXNrO6qF7UcM7pVKVPI"
    save_dir = "new_saved/testing_1/"
    file_name = "model_best.pth"

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, file_name)

    try:
        print(f"Скачивание файла с Google Диска (ID: {file_id})...")
        download_file_from_google_drive(file_id, file_path)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при скачивании файла: {e}")


if __name__ == "__main__":
    download_file()
