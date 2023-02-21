from .utils import *
from .cross_entropy_loss import *
from .resample_loss import *
# from .focal_loss import *
from .focal import *
from .afl import *

class BCEWithLogitLoss(nn.Module):
    def __init__(self, useCopyDecoupling=False):
        super(BCEWithLogitLoss, self).__init__()
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None
        self.useCopyDecoupling = useCopyDecoupling
        self.eps=1e-8
        
    def forward(self, x, y, mask=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        mask: 0 or 1 vector, same shape as y
        """
        
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if mask is not None and self.useCopyDecoupling:
            self.loss = self.loss * mask

        return -self.loss.mean()

def createLossFuntion(cfg):
    name = cfg.loss['name']
    print("using {} loss function, param is {}".format(name,cfg.loss['param']))
    if name == 'asl':
        return AsymmetricLossOptimized(**cfg.loss['param'])
    elif name == 'focal':
        return FocalLoss(**cfg.loss['param'])
    elif name == 'bce':
        return BCEWithLogitLoss(**cfg.loss['param'])
    elif name == 'afl':
        return AFLloss(**cfg.loss['param'])
    else:
        raise NotImplementedError