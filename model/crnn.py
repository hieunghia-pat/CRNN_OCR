import torch
import torch.nn as nn
from torch.nn import functional as F
from model.bi_lstm import BidirectionalLSTM

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, dropout=0.3, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'image height has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def get_predictions(self, logouts, vocab):
        ys = logouts.permute((1, 0, 2)).argmax(dim=-1)
        
        for batch_idx in range(ys.size(0)):
            idx = 0
            while ys[batch_idx, idx] != vocab.eos_idx and idx < ys.size(1)-1:
                idx += 1
            ys[batch_idx, idx+1:] = vocab.padding_idx

        return ys

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        _, _, h, _ = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return F.log_softmax(output, dim=-1)

def make_model(vocab_size, d_model=256, image_height=32, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    model = CRNN(imgH=image_height, nc=3, nclass=vocab_size, nh=d_model, dropout=dropout)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model