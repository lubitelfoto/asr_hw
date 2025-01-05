from pathlib import Path

import torch

import pandas as pd

#from torchaudio.models.decoder import ctc_decoder
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
    
        # print(f"trainer.py Модель находится на: {next(self.model.parameters()).device}")
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"Тензор {key} находится на устройстве: {value.device}")
    
        outputs = self.model(**batch)
        batch.update(outputs)
    
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
    
        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
        # Update metrics
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())
    
        for met in self.metrics["train" if self.is_train else "inference"]:
            metrics.update(met.name, met(**batch))
    
        return batch

    

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].squeeze(0).detach().cpu()

        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        #  add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly


    
        # Beam Search с использованием torchaudio
        beam_results = self.beam_search(log_probs, log_probs_length)

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.detach().cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
    
        # print(f"trainer.py Модель находится на: {next(self.model.parameters()).device}")
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"Тензор {key} находится на устройстве: {value.device}")
    
        outputs = self.model(**batch)
        batch.update(outputs)
    
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
    
        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
        # Update metrics
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())
    
        for met in self.metrics["train" if self.is_train else "inference"]:
            metrics.update(met.name, met(**batch))
    
        return batch



    def beam_search(self, log_probs, log_probs_length):
        """
        Args:
            log_probs (torch.Tensor): Тензор логарифмов вероятностей (batch_size, time, vocab_size).
            log_probs_length (torch.Tensor): Тензор длин для каждого примера в батче (batch_size,).
        
        Returns:
            list[str]: Список декодированных последовательностей.
        """
        batch_size = log_probs.size(0)
        decoded_sequences = []
    
        # Декодируем каждую запись в батче
        for i in range(batch_size):
            emissions = log_probs[i][: log_probs_length[i]].detach().cpu().numpy()  # Обрезка по длине
            top_indices = emissions.argmax(axis=-1)  # Находим индексы с максимальной вероятностью по каждому шагу времени
    
            # Используем ctc_decode для получения последовательности
            decoded_sequence = self.text_encoder.ctc_decode(top_indices)
            decoded_sequences.append(decoded_sequence)
    
        return decoded_sequences




