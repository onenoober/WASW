import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDomainLoss(nn.Module):
    def __init__(self, l1_weight=1.0, freq_weight=0.1):
        super(FrequencyDomainLoss, self).__init__()
        self.l1_weight = l1_weight
        self.freq_weight = freq_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        loss_dict = {}

        l1_loss = self.l1_loss(pred, target)
        loss_dict['l1_loss'] = l1_loss

        freq_loss = self._frequency_consistency_loss(pred, target)
        loss_dict['freq_loss'] = freq_loss

        total_loss = self.l1_weight * l1_loss + self.freq_weight * freq_loss
        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

    def _frequency_consistency_loss(self, pred, target):
        pred_freq = torch.fft.fft2(pred, norm='forward')
        target_freq = torch.fft.fft2(target, norm='forward')

        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)

        pred_phase = torch.angle(pred_freq)
        target_phase = torch.angle(target_freq)
        phase_loss = 1 - F.cosine_similarity(
            torch.stack([torch.cos(pred_phase), torch.sin(pred_phase)], dim=-1).flatten(1),
            torch.stack([torch.cos(target_phase), torch.sin(target_phase)], dim=-1).flatten(1),
            dim=1
        ).mean()

        freq_loss = magnitude_loss + 0.1 * phase_loss

        return freq_loss