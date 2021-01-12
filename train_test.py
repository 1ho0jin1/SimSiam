#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from loss import D

import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset


# kNN monitoring function
"""
1. set model as eval mode
2. calculate feature bank V from all 50000 training images >> (50000,2048)
3. calculate features of all 10000 test images (queries) >> (10000,2048)
4. calculate similarity scores using softmax >> (10000, 50000)
5. using knn (k=200), get label predictions
** when calculating feature bank, get rid of augmentation **
"""
def knn_monitor(device,model,bankloader,queryloader,k=200,t=0.1):
    model.eval()
    V = torch.zeros(50000,2048)
    Q = torch.zeros(10000,2048)
    train_label = torch.zeros(50000).type(torch.LongTensor)
    test_label = torch.zeros(10000).type(torch.LongTensor)
  
    cnt = 0
    for img, label in bankloader:
        B = img.shape[0]
        img = img.to(device)
        v = model(img,None).detach()    # detach() to save memory
        V[cnt:cnt+B] = v
        train_label[cnt:cnt+B] = label
        cnt += B
    assert cnt == 50000

    cnt = 0
    for img, label in queryloader:
        B = img.shape[0]
        img = img.to(device)
        q = model(img,None).detach()    # detach() to save memory
        Q[cnt:cnt+B] = q
        test_label[cnt:cnt+B] = label
        cnt += B
    assert cnt == 10000

    # similarity score of Q w.r.t. V
    sim_score = F.softmax(torch.mm(Q,V.T)/t,dim=1)    # (10000,50000)

    # knn : k features with highest similarity
    # torch.topk : returns val, idx >> get idx: index of training sample w/ highest similarity
    # torch.mode : returns val, idx >> get val: most voted label
    _, k_idx = torch.topk(sim_score,k)    # (10000,k)
    pred, _ = torch.mode(train_label[k_idx],dim=1)    # 10000

    # get back to train mode
    model.train() 

    return torch.sum(pred == test_label), len(test_label)



def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']



def save_chkpt(model,optimizer,scheduler,epoch,loss_hist,knn_hist,save_dir):
    state_dict = {
        "model_state_dict":model.state_dict(),
        "optim_state_dict":optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict(),
        "epoch":epoch,
        "history":(loss_hist,knn_hist)
    }
    torch.save(state_dict,save_dir)


    
def train_SimSiam(device,trainloader,bankloader,queryloader,model,optimizer,scheduler,num_epochs,base_dir,
                  saved_epoch=0,loss_hist=[],knn_hist=[],best_acc=80):
    start_time = time.time()
    for epoch in range(saved_epoch,num_epochs):
        model.train()
        epoch_time = time.time()

        for b,(x1,x2,label,_) in enumerate(trainloader):
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad()
            z1,z2,p1,p2 = model(x1,x2)
            loss = D(p1,z2)/2 + D(p2,z1)/2
            loss.backward()
            optimizer.step()

            if b%40 == 0:
                loss_hist.append(loss.item())
                now = datetime.datetime.now()
                print("epoch:{} batch:{} loss:{:.4f} lr:{:.2e} time:{}".format(epoch,b,loss.item(),get_lr(optimizer),
                                                                               now.strftime('%Y-%m-%d %H:%M:%S')))

        # adjust lr by cosine annealing
        scheduler.step()
        correct,total = knn_monitor(device,model,bankloader,queryloader)
        accuracy = correct/total*100
        knn_hist.append(accuracy)
        print("epoch:{} knn_acc:{:.2f}% ({}/{}) epoch_time:{:.2f}\n".format(epoch,accuracy,correct,total,
                                                                            time.time()-epoch_time))
        if accuracy > best_acc:
            save_dir = "SimSiam_pre_{}.chkpt".format(int(accuracy*100))
            save_chkpt(model,optimizer,scheduler,epoch,loss_hist,knn_hist,base_dir+save_dir)
            best_acc = accuracy
            print("---------------- model saved: {:.2f}% ----------------".format(accuracy))
    print("training finished!\ntotal time:{:.2f}, best_acc:{:.2f}".format(time.time()-start_time,best_acc))

    

def train_classifier(device,classifier_trainloader,queryloader,model,classifier,optimizer,scheduler,num_epochs,base_dir,best_acc=80):
    for epoch in range(num_epochs):
        model.eval()
        epoch_time = time.time()
        if get_lr(optimizer) < 1e-3:
            print("training finished due to accuracy stall")
            break

        for batch, (img,label) in enumerate(classifier_trainloader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                feature = model(img,None)
            pred = classifier(feature)
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()
            if batch%40 == 0:
                now = datetime.datetime.now()
                print("epoch:{} batch:{} loss:{:.4f} lr:{:.2e} time:{}".format(epoch, batch,loss.item(),get_lr(optimizer),
                                                                               now.strftime('%Y-%m-%d %H:%M:%S')))

            correct = 0
            with torch.no_grad():
                for batch, (img,label) in enumerate(queryloader):
                    img,label = img.to(device), label.to(device)
                    feature = model(img,None)
                    pred = classifier(feature)
                    correct += torch.sum(torch.argmax(pred,dim=1) == label)
                accuracy = correct/100.
                print("epoch:{} accuracy:{:.2f}% ({}/{})\n".format(epoch,accuracy,correct,10000))

            scheduler.step(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(classifier.state_dict(),base_dir+"classifier_best.pth")
                print("---------------- classifier saved: {:.2f}% ----------------\n".format(accuracy))

