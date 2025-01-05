from torch import nn
from torch.nn import Sequential


class DeepSpeech2Model(nn.Module):
    def __init__(self, input_dim=80, n_tokens=29):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.rnn = None
        self.fc = nn.Linear(128 * 2, n_tokens)

    def _initialize_rnn(self, input_dim):
        """
        Инициализация RNN после расчёта реального размера выхода.
        """
        self.rnn_input_dim = input_dim
        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        # Перемещение RNN на то же устройство, что и остальные параметры модели
        self.rnn = self.rnn.to(next(self.parameters()).device)

    @staticmethod
    def calculate_conv_output_dim(input_dim, kernel_size, stride, padding):
        return (input_dim + 2 * padding - kernel_size) // stride + 1

    def forward(self, spectrogram, spectrogram_length, **kwargs):
        """
        Прямой проход через модель. Обрабатывает входной спектрограммы и возвращает предсказания
        и длины выходных данных.
        
        Args:
            spectrogram (torch.Tensor): Входная спектрограмма [batch_size, time, freq].
            spectrogram_length (torch.Tensor): Длины спектрограмм в пакете.
    
        Returns:
            dict: Словарь с выходными данными:
                - "log_probs": Логарифмические вероятности (torch.Tensor).
                - "input_lengths": Длины выходов после обработки (torch.Tensor).
        """
        # Удаляем лишнюю размерность, если spectrogram имеет 4 измерения
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.squeeze(1)
    
        # Добавляем канал и пропускаем через свёрточный слой
        x = spectrogram.unsqueeze(1)  # [batch_size, 1, time, freq]
        x = self.conv(x)
    
        # Получаем размеры выхода после свёртки
        batch_size, channels, height, width = x.shape
        if self.rnn is None:  # Инициализация RNN при первом вызове
            self._initialize_rnn(height * channels)
    
        # Вычисляем длины входов после свёрточных слоёв
        input_lengths = self.calculate_conv_output_dim(
            spectrogram_length, kernel_size=11, stride=2, padding=5
        )
        input_lengths = self.calculate_conv_output_dim(
            input_lengths, kernel_size=11, stride=1, padding=5
        )
    
        # Перестановка осей для подачи в RNN
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, time, channels, features]
        x = x.view(x.size(0), x.size(1), -1)  # [batch_size, time, features]
    
        # Пропуск через RNN и полносвязный слой
        x, _ = self.rnn(x)
        log_probs = self.fc(x)
    
        # Возврат результата
        return {
            "log_probs": log_probs,         # Логарифмические вероятности
            "log_probs_length": input_lengths,  # Длины выходов после обработки
        }





    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


    def check_cpu_tensors(self):
        """
        Печатает параметры и буферы модели, которые находятся на CPU.
        """
        print("deepspeech2 Checking tensors on CPU...")
        for name, param in self.named_parameters():
            if param.device.type == "cpu":
                print(f"Parameter '{name}' is on CPU with shape {param.shape}")
        
        for name, buffer in self.named_buffers():
            if buffer.device.type == "cpu":
                print(f"Buffer '{name}' is on CPU with shape {buffer.shape}")

        print("deepspeech2 Finished checking CPU tensors.")
