#!/usr/bin/env python
# coding: utf-8

# In[2]:


from models import SimSiam_CIFAR
from dataset import CIFAR10_
from loss import D
from train_test import knn_monitor, get_lr, save_chkpt, train_SimSiam, train_classifier
from ECE import calib_metrics, ModelWithTemperature, _ECELoss

import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset


base_dir = ""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
label_dict = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck"
}

# CIFAR10 mean, std
CIFAR_mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
CIFAR_std = torch.FloatTensor([0.2023, 0.1994, 0.2010])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_mean, std=CIFAR_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_mean, std=CIFAR_std),
])

# for SimSiam training
train_batch_size = 512
trainset = CIFAR10_(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True)

# for knn monitoring
bank_batch_size = 500
query_batch_size = 100
bankset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
queryset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
bankloader = torch.utils.data.DataLoader(bankset, batch_size=bank_batch_size, shuffle=False, drop_last=False)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=query_batch_size, shuffle=False, drop_last=False)

# for classifier training : must shuffle trainset
classifier_trainloader = torch.utils.data.DataLoader(bankset,batch_size=bank_batch_size,shuffle=True,drop_last=False)



# SimSiam configure
model = SimSiam_CIFAR().to(device)
num_epochs = 800
lr = 0.03
momentum = 0.9
weight_decay = 0.0005
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs)

# SimSiam training
train_SimSiam(device=device, trainloader=trainloader, bankloader=bankloader, queryloader=queryloader,
              model=model, optimizer=optimizer, scheduler=scheduler, num_epochs = num_epochs,
              base_dir=base_dir,best_acc=0)


# In[ ]:




