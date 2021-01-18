import torch
import torch.nn as nn
import torch.optim as optim

from config import config, preprocess
from models import SimSiam_CIFAR
from train import _train_SimSiam, _train_classifier


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args = config()


def train_SimSiam():
    trainloader,bankloader,queryloader,_ = preprocess(args)
    model = SimSiam_CIFAR()
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_epochs)

    _train_SimSiam(device=device,trainloader=trainloader,bankloader=bankloader,queryloader=queryloader,
                  model=model,optimizer=optimizer,scheduler=scheduler,num_epochs=args.num_epochs,
                  base_dir=args.save_dir,best_acc=args.save_acc)



def train_classifier(weight_path, classifier_type='lin_small'):
    """
    :param classifier_type:
    - 'lin_small' >> Default, simplest linear classifier
    - 'lin_large' >> large linear classifier without nonlinearity
    - 'nonlin_large' >> large linear classifier with nonlinearity; solely for performance
    """
    _,_,queryloader,classifier_trainloader = preprocess(args)

    model = SimSiam_CIFAR()
    chkpt = torch.load(weight_path)
    model.load_state_dict(chkpt['model_state_dict'])

    if classifier_type == 'lin_small':
        classifier = nn.Linear(2048,10)

    elif classifier_type == 'lin_large':
        classifier = nn.Sequential(nn.Linear(2048,2048),
                                   nn.Linear(2048,10))

    elif classifier_type == 'nonlin_large':
        classifier = nn.Sequential(nn.Linear(2048,2048),
                                   nn.BatchNorm1d(2048),
                                   nn.ReLU(),
                                   nn.Linear(2048,10))

    optimizer = optim.SGD(classifier.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.1,patience=10)
    criterion = nn.CrossEntropyLoss()

    _train_classifier(device=device,criterion=criterion,classifier_trainloader=classifier_trainloader,
                     queryloader=queryloader,model=model,classifier=classifier,optimizer=optimizer,scheduler=scheduler,
                     num_epochs=200,base_dir=args.save_dir,best_acc=args.save_acc)

# SimSiam module training
train_SimSiam()

# classifier training
train_classifier('./weights/SimSiam_8812.chkpt', classifier_type='lin_small')
