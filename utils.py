import torch
import torch.nn.functional as F

# kNN monitoring function
def knn_monitor(device,model,bankloader,queryloader,k=200,t=0.1):
    """
    1. set model as eval mode
    2. calculate feature bank V from all 50000 training images >> (50000,2048)
    3. calculate features of all 10000 test images (queries) >> (10000,2048)
    4. calculate similarity scores using softmax >> (10000, 50000)
    5. using knn (k=200), get label predictions
    ** when calculating feature bank, get rid of augmentation **
    """
    model.eval()
    with torch.no_grad():
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
        _, k_idx = torch.topk(sim_score,k)    # (10000,k)
        pred, _ = torch.mode(train_label[k_idx],dim=1)    # 10000

    # get back to train mode
    model.train()

    return torch.sum(pred == test_label).item(), len(test_label)



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