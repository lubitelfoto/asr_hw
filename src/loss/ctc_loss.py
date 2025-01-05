import torch
from torch import Tensor
from torch.nn import CTCLoss

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
    
        # if not isinstance(log_probs_length, torch.Tensor):
        #     log_probs_length = torch.tensor(log_probs_length, dtype=torch.int32)
        # if not isinstance(text_encoded_length, torch.Tensor):
        #     text_encoded_length = torch.tensor(text_encoded_length, dtype=torch.int32)
      
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}

