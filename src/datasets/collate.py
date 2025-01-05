import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.
    """
    # Собираем текстовые данные

    #print(f"collate {dataset_items}")

    raw_text = [item['text']  for item in dataset_items]
    
    text = [item['text_encoded'].squeeze(0) if item['text_encoded'].dim() > 1 else item['text_encoded'] for item in dataset_items]
    text_lengths = [t.size(0) for t in text]  # Сохраняем длины текстов
    max_text_length = max(text_lengths)
    text_padded = torch.stack([
        torch.cat([t, torch.zeros(max_text_length - t.size(0))]) for t in text
    ])

    # Остальные данные
    audio = [item['audio'].squeeze(0) if item['audio'].dim() > 1 else item['audio'] for item in dataset_items]
    max_audio_length = max([a.size(0) for a in audio])
    audio_padded = torch.stack([torch.cat([a, torch.zeros(max_audio_length - a.size(0))]) for a in audio])

    spectrograms = [item['spectrogram'] for item in dataset_items]
    spectrogram_lengths = [s.size(-1) for s in spectrograms]
    max_spectrogram_length = max(spectrogram_lengths)
    spectrograms_padded = torch.stack([
        torch.nn.functional.pad(s, (0, max_spectrogram_length - s.size(-1))) for s in spectrograms
    ])

    audio_paths = [item['audio_path'] for item in dataset_items]

    # Формируем итоговый батч
    result_batch = {
        'text': raw_text,
        'text_encoded': text_padded,
        'text_encoded_length': torch.tensor(text_lengths, dtype=torch.int32),  # Исправлено
        'audio': audio_padded,
        'spectrogram': spectrograms_padded,
        'spectrogram_length': torch.tensor(spectrogram_lengths, dtype=torch.int32),  # Оставляем как есть
        'audio_path': audio_paths,
    }

    return result_batch


# def collate_fn(dataset_items: list[dict]):
#     """
#     Collate and pad fields in the dataset items.
#     Converts individual items into a batch.
#     """
#     # Собираем текстовые данные
#     text = [item['text_encoded'].squeeze(0) if item['text_encoded'].dim() > 1 else item['text_encoded'] for item in dataset_items]
#     max_text_length = max([t.size(0) for t in text])
#     text_padded = torch.stack([
#         torch.cat([t, torch.zeros(max_text_length - t.size(0))]) for t in text
#     ])

#     # Остальные данные
#     audio = [item['audio'].squeeze(0) if item['audio'].dim() > 1 else item['audio'] for item in dataset_items]
#     max_audio_length = max([a.size(0) for a in audio])
#     audio_padded = torch.stack([torch.cat([a, torch.zeros(max_audio_length - a.size(0))]) for a in audio])

#     spectrograms = [item['spectrogram'] for item in dataset_items]
#     spectrogram_lengths = [s.size(-1) for s in spectrograms]
#     max_spectrogram_length = max(spectrogram_lengths)
#     #max_spectrogram_length = max([s.size(-1) for s in spectrograms])
#     spectrograms_padded = torch.stack([
#         torch.nn.functional.pad(s, (0, max_spectrogram_length - s.size(-1))) for s in spectrograms
#     ])

#     audio_paths = [item['audio_path'] for item in dataset_items]

#     # Формируем итоговый батч
#     result_batch = {
#         'text_encoded': text_padded,
#         'text_encoded_length': max_text_length,
#         'audio': audio_padded,
#         'spectrogram': spectrograms_padded,
#         'spectrogram_length': torch.tensor(spectrogram_lengths),
#         'audio_path': audio_paths,
#     }

#     return result_batch


