# SimSiam
Pytorch implementation of [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) by Xinlei Chen & Kaiming He

####CIFAR-10 Accuracy  
&nbsp;|**KNN Acc (%)** | **Linear Eval (%)**    
------------|:---:|:---:
**Paper**|-|91.8
**This Repo**|88.44|87.30  

Command Line Arguments
* train_batch_size (default:512)
* num_epochs (default:800)
* lr (default:0.03)
* momentum (default:0.9)
* weight_decay (default:0.0005)
* save_dir (default:current directory)
* save_acc (default:80%)  
Note: the weight file starts to save when knn accuracy surpasses save_acc