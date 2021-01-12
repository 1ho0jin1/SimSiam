#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class calib_metrics():
    def __init__(self,targets,dataloader,model):
        self.targets = targets
        self.logits = self.get_logits(dataloader,model)


    def get_logits(self,dataloader,model):
        N = len(self.targets)
        logits = torch.zeros(N,10)
        with torch.no_grad():
            n = 0
            for img,label in dataloader:
                img,label = img.to(device),label.to(device)
                b = len(label)
                logits[n:n+b] = model(img)
                n+=b
            assert n == N
        return logits


    def make_bins(self,M):
        bins = {}
        prev = 0
        conf,pred = torch.max(F.softmax(self.logits,dim=1),dim=1)
        for m,i in enumerate(torch.linspace(0,1,M+1)):
            if m==0: continue
            idx = (conf>=prev)*(conf<i)
            bins[m-1] = torch.where(idx==True)[0]
            prev = i
        return bins


    def ECE(self,M):
        ece = 0
        conf,pred = torch.max(F.softmax(self.logits,dim=1),dim=1)
        bins = self.make_bins(M)
        for m in range(M):
            indices = bins[m]
            bin_size = len(indices)
            if bin_size == 0: continue
            bin_acc = torch.sum(pred[indices] == self.targets[indices])
            bin_conf = torch.sum(conf[indices])
            ece += torch.abs(bin_acc-bin_conf)
        ece /= len(self.targets)
        return ece


    def ACC(self,M):
        acc_hist = []
        _,pred = torch.max(F.softmax(self.logits,dim=1),dim=1)
        bins = self.make_bins(M)
        for m in range(M):
            indices = bins[m]
            bin_size = len(indices)
            if bin_size == 0:
                acc_hist.append(0)
                continue
            acc = torch.sum(pred[indices] == self.targets[indices])/bin_size
            acc_hist.append(acc.item())
        return acc_hist

    def reliability_diagram(self,M):
        acc_hist = self.ACC(M)
        plt.figure(figsize=(4,4))
        plt.bar(torch.arange(0,M)+0.5,acc_hist)
        plt.xticks(torch.arange(0,M+1),["{:.2f}".format(i/M) for i in range(M+1)],rotation="vertical")
        plt.title("Reliability Diagram (M={})".format(M))
        plt.xlabel("confidence")
        plt.ylabel("accuracy")
        plt.show()
        
        
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

