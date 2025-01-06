from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class MyCustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_file, *args, **kwargs):
        transcriptions = self._load_transcriptions(transcription_file)
        data = []

        for path in Path(audio_dir).iterdir():
            entry = {}

            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)

                audio_tensor, sr = torchaudio.load(entry["path"])
                entry["audio_len"] = audio_tensor.shape[1] / sr

                file_stem = path.stem
                if file_stem in transcriptions:
                    entry["text"] = transcriptions[file_stem]
                else:
                    print(f"Транскрипция для {file_stem} отсутствует.")

            if "path" in entry and "audio_len" in entry and "text" in entry:
                data.append(entry)

        super().__init__(data, *args, **kwargs)

    @staticmethod
    def _load_transcriptions(transcription_file):
        transcriptions = {}
        with open(transcription_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    print(f"Пропущена строка: {line.strip()}")
                    continue
                file_name, text = parts
                transcriptions[file_name] = text
        return transcriptions
