from torch import nn
from torch.nn import Sequential


class DeepSpeech2Model(nn.Module):
    def __init__(self, input_dim=128, n_tokens=29):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Рассчитываем размер выхода сверточной части
        conv_output_dim = self._calculate_conv_output_dim(input_dim)

        # Инициализируем RNN
        self.rnn = nn.GRU(
            input_size=conv_output_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(128 * 2, n_tokens)

    def _calculate_conv_output_dim(self, input_dim):
        """
        Рассчитывает размер выхода сверточной части.
        """
        freq = input_dim
        freq = self.calculate_conv_output_dim(
            freq, kernel_size=41, stride=2, padding=20
        )
        freq = self.calculate_conv_output_dim(
            freq, kernel_size=21, stride=2, padding=10
        )
        return freq * 32  # 32 — количество каналов

    @staticmethod
    def calculate_conv_output_dim(input_dim, kernel_size, stride, padding):
        return (input_dim + 2 * padding - kernel_size) // stride + 1

    def forward(self, spectrogram, spectrogram_length, **kwargs):
        """
        Args:
            spectrogram (torch.Tensor): Входная спектрограмма [batch_size, time, freq].
            spectrogram_length (torch.Tensor): Длины спектрограмм в пакете.

        Returns:
            dict: Словарь с выходными данными:
                - "log_probs": Логарифмические вероятности (torch.Tensor).
                - "input_lengths": Длины выходов после обработки (torch.Tensor).
        """

        # print(f"deepspeech2.py spectrogram.shape {spectrogram.shape} spectrogram_length {spectrogram_length.shape}")

        if spectrogram.dim() == 4:
            spectrogram = spectrogram.squeeze(1)

        x = spectrogram.unsqueeze(1)
        x = self.conv(x)

        batch_size, channels, height, width = x.shape

        input_lengths = self.calculate_conv_output_dim(
            spectrogram_length, kernel_size=11, stride=2, padding=5
        )
        input_lengths = self.calculate_conv_output_dim(
            input_lengths, kernel_size=11, stride=1, padding=5
        )

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.rnn(x)
        log_probs = self.fc(x)

        return {
            "log_probs": log_probs,
            "log_probs_length": input_lengths,
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
