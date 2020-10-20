from torchsummary import summary
import torch.nn as nn

class Resnet50(nn.Module):
    def __init__(self, num_classes, input_size):
        super(Resnet50, self).__init__()
        


model = Resnet50().cuda()
summary(model, ())