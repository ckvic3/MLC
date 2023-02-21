import torch
import torch.nn as nn

class AFLloss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, factor, num_classes = 20, eps=1e-8, beta = 0.1, disable_torch_grad_focal_loss=False,useCopyDecoupling=False):
        super(AFLloss, self).__init__()

        self.factor = factor
        self.beta = beta
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None
        self.useCopyDecoupling = useCopyDecoupling
        self.gamma_pos = torch.ones(num_classes).cuda() * 2
        self.gamma_neg = torch.ones(num_classes).cuda() * 2

    def forward(self, x, y, mask=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        
        self.targets = y
        self.anti_targets = 1 - y


        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        self.gamma_pos = self.factor / (self.xs_neg.detach() + self.beta)
        self.gamma_neg = self.factor / (self.xs_pos.detach() + self.beta) 

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        
        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w
        else:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                        self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
            self.loss *= self.asymmetric_w  

        if mask is not None and self.useCopyDecoupling:
            self.loss = self.loss * mask

        return -self.loss.mean()




