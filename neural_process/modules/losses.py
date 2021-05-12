from torch import nn


class ConditionalLoss(nn.Module):
    @staticmethod
    def forward(log_likelihood, x_target):
        num_target_points = x_target.shape[1]
        loss = -log_likelihood
        normalized_loss = loss / num_target_points
        return normalized_loss


class LatentLoss(nn.Module):
    @staticmethod
    def forward(log_likelihood, kl_divergence, x_target):
        num_target_points = x_target.shape[1]
        loss = -log_likelihood + kl_divergence
        normalized_loss = loss / num_target_points
        return normalized_loss
