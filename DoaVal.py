
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
import DOADataSets, DOAmodel
import numpy as np
import torch

import matplotlib.pyplot as plt
# 1. 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}", flush=True)

# len是函数 永远返回dim0的大小， shape是属性  size 是函数
angleGird = np.linspace(-90, 90, 181)

# 加载数据
doaDataSet = DOADataSets.DoaDataSet('./dataCsv.csv', './labelCsv.csv')

# 只取 1/20 的样本（先跑通。如果要全量，把 // 20 删掉即可）
small_size = len(doaDataSet) 
small_dataset = Subset(doaDataSet, range(small_size))

print(f"total: {len(doaDataSet)}, using: {len(small_dataset)}", flush=True)

# 再按 8:2 划分
train_size = int(0.8 * len(small_dataset))
test_size = len(small_dataset) - train_size
train_dataset, test_dataset = random_split(small_dataset, [train_size, test_size])


batch_size = 2048


testLd = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=2, pin_memory=True
)



from sklearn.metrics import accuracy_score, hamming_loss, f1_score
def getMaxAcc():
    doaModel = torch.load('full_myFnn_model.pth')
    doaModel = doaModel.to(device)
    ths = np.linspace(0,1,200)
    doaModel.eval()
    all_probs = []
    all_labels =[]

    with torch.no_grad():
        for feas,labels in testLd:
            feas = feas.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = doaModel(feas)
            probs = torch.sigmoid(logits)
       

           
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
                # print("Subset Accuracy:", accuracy_score(labels,pred))
    all_probs = torch.cat(all_probs,0).numpy()
    all_labels = torch.cat(all_labels,0).numpy()

    if np.issubdtype(all_labels.dtype, np.floating):
        all_labels = (all_labels > 0.5).astype(int)

    best_th = 0.5
    best_acc = -1.0
    acc_list =[]
    ths = np.linspace(0, 1, 200)
    for th in ths:
        pred = (all_probs >= th).astype(int)
        acc = accuracy_score(all_labels, pred)   # Subset Accuracy
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_th = th
    plt.figure()
    plt.plot(ths,acc_list)
    plt.show()
    print(f'best th{best_th} best acc{best_acc}')



    return best_th,best_acc
getMaxAcc()