import torch
import torch.optim as optim

class resnet18(torch.nn.Module):
    """ Example Model. ResNet18 with a regression layer on top. """
    def __init__(self, n_classes, num_labels, mode='regress'):
        super(resnet18, self).__init__()
        self.base_model = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
        self.mode = mode
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)
        if(mode == 'regress'):
            self.fc = torch.nn.Linear(base_model_fc_size, num_labels)
        elif(mode == 'classify'):
            self.fc = torch.nn.Linear(base_model_fc_size, n_classes)
        self.sm = torch.nn.Softmax()

    def forward(self, images):
        im_features = self.base_model(images)
        preds = self.fc(im_features.squeeze())
        return preds

class cnn3(torch.nn.Module):
    """ small network trained from scratch"""
    def __init__(self, n_classes, num_labels, mode='regress'):
        super(cnn3, self).__init__()
        # convolutional layers
        self.conv1 = torch.nn.Conv2d(3,   5, 5, stride=2)
        self.conv2 = torch.nn.Conv2d(5,  10, 5, stride=2)
        self.conv3 = torch.nn.Conv2d(10, 20, 5, stride=2)
        self.relu  = torch.nn.ReLU()

        self.mode = mode

        if(mode == 'classify'):
            self.fc1  = torch.nn.Linear(6480, n_classes)
        elif(mode == 'regress'):
            self.fc1  = torch.nn.Linear(6480, num_labels)

        self.sm  = torch.nn.Softmax()

    def forward(self, images):
        im_size = images.shape
        h1 = self.relu(self.conv1(images))
        h2 = self.relu(self.conv2(h1))
        h3 = self.relu(self.conv3(h2))

        preds = self.fc1(h3.view(im_size[0], -1).squeeze())
        return preds

class cnn5(torch.nn.Module):
    """ small network trained from scratch"""
    def __init__(self, n_classes, num_labels, mode='regress'):
        super(cnn5, self).__init__()
        # convolutional layers
        self.conv1 = torch.nn.Conv2d(3,   5, 5, padding=True)
        self.conv2 = torch.nn.Conv2d(5,  10, 5, padding=True)
        self.conv3 = torch.nn.Conv2d(10, 20, 5, padding=True)
        self.conv4 = torch.nn.Conv2d(20, 30, 5, padding=True)
        self.conv5 = torch.nn.Conv2d(30, 40, 5, padding=True)
        self.pool  = torch.nn.AvgPool2d(2, stride=2)
        #self.relu  = torch.nn.ReLU()

        self.mode = mode

        if(mode == 'classify'):
            self.fc1  = torch.nn.Linear(360, n_classes)
        elif(mode == 'regress'):
            self.fc1  = torch.nn.Linear(360, num_labels)

        #self.fc2 = torch.nn.Linear(100,  num_labels)
        self.sm   = torch.nn.Softmax()


    def forward(self, images):
        im_size = images.size()

        h1 = F.relu(self.pool(self.conv1(images)))
        h2 = F.relu(self.pool(self.conv2(h1)))
        h3 = F.relu(self.pool(self.conv3(h2)))
        h4 = F.relu(self.pool(self.conv4(h3)))
        h5 = F.relu(self.pool(self.conv5(h4)))

        if(self.mode == 'classify'):
            preds = self.fc1(h5.view(im_size[0] , -1).squeeze())

        elif(self.mode == 'regress'):
            preds = self.fc1(h5.view(im_size[0], -1).squeeze()) #OPT: add softmax
            #preds = self.fc2(self.fc1(h5.view(im_size[0], n_classes, -1).squeeze()))

        return preds
