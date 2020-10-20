from torchsummary import summary
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes, input_size):
        super(VGG16, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),
            #------------------------------
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),
            #------------------------------
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),
            #------------------------------
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),
            #------------------------------
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0)
        )

        self.fc = nn.Sequential(
            nn.Linear(((int(input_size)//(2**5))**2)*512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes),
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        predict = self.fc(x)
        return predict

model = VGG16(num_classes = 1000, input_size = 224).cuda()
summary(model, (3, 224, 224))