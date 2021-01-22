import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dataset import CIFAR10_


def config():
    parser = argparse.ArgumentParser(description='SimSiam_CIFAR')
    parser.add_argument('--train_batch_size',default=512,type=int)
    parser.add_argument('--num_epochs',default=800,type=int,help='total training epochs')
    parser.add_argument('--lr',default=3e-2,type=float,help='initial learning rate')
    parser.add_argument('--momentum',default=0.9,type=float,help='momentum')
    parser.add_argument('--weight_decay',default=5e-4,type=float,help='weight_decay')
    parser.add_argument('--save_dir',default='./',type=str,help='checkpoint save directory')
    parser.add_argument('--save_acc',default=80,type=int,help='save weight when above this accuracy')
    args = parser.parse_args()

    return args


def preprocess(args):
    # CIFAR10 mean, std
    CIFAR_mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
    CIFAR_std = torch.FloatTensor([0.2023, 0.1994, 0.2010])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
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

    train_batch_size = args.train_batch_size
    bank_batch_size = 500
    query_batch_size = 100

    trainset = CIFAR10_(root='./data',train=True,download=True,transform=transform_train)
    bankset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    queryset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset,batch_size=train_batch_size,shuffle=True,num_workers=4,pin_memory=False)
    bankloader = DataLoader(bankset,batch_size=bank_batch_size,shuffle=False)
    queryloader = DataLoader(queryset,batch_size=query_batch_size,shuffle=False)
    classifier_trainloader = DataLoader(bankset,batch_size=bank_batch_size,shuffle=True)

    return trainloader,bankloader,queryloader,classifier_trainloader


