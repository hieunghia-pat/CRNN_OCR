import torch
import torch.nn as nn

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class CTCLoss(nn.Module):
    def __init__(self, padding_idx) -> None:
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(blank=padding_idx, reduction="mean", zero_infinity=True)

    def forward(self, x, target, x_len, target_len):
        return self.criterion(x, target, x_len, target_len)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, y_len):
        T, N, _ = x.shape
        x_len = [[T]*N]
        x_len = torch.tensor(x_len).long().cuda()
        loss = self.criterion(x, y, x_len, y_len)
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss