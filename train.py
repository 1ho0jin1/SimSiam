import time
import datetime
import torch

from loss import D
from utils import knn_monitor,get_lr,save_chkpt


def _train_SimSiam(device,trainloader,bankloader,queryloader,
                   model,optimizer,scheduler,num_epochs,base_dir,
                  saved_epoch=0,loss_hist=[],knn_hist=[],best_acc=80):
    model.to(device)
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
            save_dir = base_dir+"SimSiam_{}.chkpt".format(int(accuracy*100))
            save_chkpt(model,optimizer,scheduler,epoch,loss_hist,knn_hist,base_dir+save_dir)
            best_acc = accuracy
            print("---------------- model saved: {:.2f}% ----------------".format(accuracy))
    print("training finished!\ntotal time:{:.2f}, best_acc:{:.2f}".format(time.time()-start_time,best_acc))

    

def _train_classifier(device,criterion,classifier_trainloader,queryloader,
                      model,classifier,optimizer,scheduler,num_epochs,base_dir,best_acc=80):
    model.to(device)
    classifier.to(device)
    for epoch in range(num_epochs):
        model.eval()
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
                correct += torch.sum(torch.argmax(pred,dim=1) == label).item()
            accuracy = correct/100    # since CIFAR10 testset has 10000 samples
            print("epoch:{} accuracy:{:.2f}% ({}/{}) lr:{:.2e}\n".format(epoch,accuracy,correct,10000,get_lr(optimizer)))

        scheduler.step(accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(classifier.state_dict(),base_dir+"classifier_best.pth")
            print("---------------- classifier saved: {:.2f}% ----------------\n".format(accuracy))

