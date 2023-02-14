import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch


__all__ = [
    'SelVGG', 'sel_vgg16', 'sel_vgg16_bn'
]


class SelVGG(nn.Module):


    def __init__(self, features, selective=True, output_dim=1000, input_size = 32):
        super(SelVGG, self).__init__()
        self.features = features
        self.selective = selective
        self.output_dim = output_dim
        if self.selective==False:
            if input_size == 32:
                self.classifier = nn.Sequential(nn.Linear(512,512), nn.ReLU(inplace=True), \
                                            nn.BatchNorm1d(512),nn.Dropout(0.5),nn.Linear(512, self.output_dim))
            elif input_size == 64:
                self.classifier = nn.Sequential(nn.Linear(2048,512), nn.ReLU(inplace=True), \
                                            nn.BatchNorm1d(512),nn.Dropout(0.5),nn.Linear(512, self.output_dim))
        else:
            if input_size==32:    
                self.dense_class = torch.nn.Linear(512, self.output_dim)
                self.dense_selec_1 = torch.nn.Linear(512, 512)
                self.dense_auxil = torch.nn.Linear(512, self.output_dim)
                self.batch_norm = torch.nn.BatchNorm1d(512)
                self.dense_selec_2 = torch.nn.Linear(512,1)
            else:
                self.dense_class = torch.nn.Linear(2048, self.output_dim)
                self.dense_selec_1 = torch.nn.Linear(2048, 512)
                self.dense_auxil = torch.nn.Linear(2048, self.output_dim)
                self.batch_norm = torch.nn.BatchNorm1d(512)
                self.dense_selec_2 = torch.nn.Linear(512,1)
                                                    
        self._initialize_weights()

        



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.selective:
            h = self.dense_class(x)
            h = torch.nn.functional.softmax(h, dim=1)
            g = self.dense_selec_1(x)
            g = torch.nn.functional.relu(g)
            g = self.batch_norm(g)
            g = self.dense_selec_2(g)
            g = torch.sigmoid(g)
            a = self.dense_auxil(x)
            a = torch.nn.functional.softmax(a, dim=1)
            hg = torch.cat([h,g],1)
            return hg, a
        else:
            x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v)==int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout(v)]
    return nn.Sequential(*layers)


cfg = {
    'D': [64,0.3, 64, 'M', 128,0.4, 128, 'M', 256,0.4, 256,0.4, 256, 'M', 512,0.4, 512,0.4, 512, 'M', 512,0.4, 512,0.4, 512, 'M',0.5],
    'D2': [64,0.15, 64, 'M', 128,0.2, 128, 'M', 256,0.2, 256,0.2, 256, 'M', 512,0.2, 512,0.2, 512, 'M', 512,0.2, 512,0.2, 512, 'M',0.25]
}

def sel_vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SelVGG(make_layers(cfg['D']), **kwargs)
    return model


def sel_vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = SelVGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

    
