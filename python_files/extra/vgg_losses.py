import torch
import torchvision
from constants import LAMBDA_P, LAMBDA_S


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()

        vgg19 = torchvision.models.vgg19_bn()
        vgg19.load_state_dict(torch.load('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Base/vgg19_bn-c79401a0.pth'))
        vgg19.eval()
        blocks = [
            vgg19.features[:3],
            vgg19.features[3:10],
            vgg19.features[10:17],
            vgg19.features[17:30],
            vgg19.features[30:43]
        ]

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inputs, targets, feature_layers=(2, 3), style_layers=(1, 2, 3, 4)):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = (inputs - self.mean) / self.std
        targets = (targets - self.mean) / self.std
        inputs = self.transform(inputs, mode='bilinear', size=(224, 224), align_corners=False)
        targets = self.transform(targets, mode='bilinear', size=(224, 224), align_corners=False)
        perc_loss = 0.0
        style_loss = 0.0
        x = inputs
        y = targets

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                perc_loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        perc_loss *= 1 / len(feature_layers)
        style_loss *= 1 / len(style_layers)
        return perc_loss, style_loss
