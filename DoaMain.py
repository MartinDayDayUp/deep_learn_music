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

# 2. 模型搬到 GPU
doaModel = DOAmodel.DOAModel(angleGird.shape[0], 2).to(device)

batch_size = 4096
epochs = 1

# 3. DataLoader 优化（num_workers 和 pin_memory 能加速 GPU 训练）
trainLd = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, 
    num_workers=2, pin_memory=True
)
testLd = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=2, pin_memory=True
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(doaModel.parameters(), lr=1e-4)



# def Train():
#     doaModel.train()
#     for ep in range(epoch):
#         train_loss = 0.0

#         for feas, labels in trainLd:
#             # 4. 数据搬到 GPU（non_blocking 配合 pin_memory 加速传输）
#             feas = feas.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             logits = doaModel(feas)
#             loss = criterion(logits, labels)

#             optimizer.zero_grad()  # 清空梯度
#             loss.backward()        # 反向传播
#             optimizer.step()       # 更新参数

#             train_loss += loss.item() * len(feas)

#         train_loss = train_loss / len(trainLd.dataset)
#         print(f'Epoch: {ep+1} Training Loss: {train_loss:.6f}', flush=True)

# import matplotlib.pyplot as plt
# def Test(isOneTest):
#     if isOneTest:
#         doaModel.eval()

#         # 1. 取测试集里的第 5 个样本
#         feas, label = test_dataset[5]

#         # 2. 加 batch 维度，搬到 GPU
#         feas = feas.unsqueeze(0).to(device)      # (1, 2, 8, 8)

#         # 3. 前向 + Sigmoid 得到概率
#         with torch.no_grad():
#             logits = doaModel(feas)              # (1, 181)
#             prob = torch.sigmoid(logits)         # (1, 181)

#         # 4. 真实标签中 > 0 的索引
#         true_idx = torch.nonzero(label > 0).flatten()       # (K,)
#         plt.plot(angleGird,prob)
#         # 5. 预测概率 > 0.65 的索引
#         pred_idx = torch.where(prob[0] > 0.65)[0]           # (M,)

#         # 6. 打印
#         print(f"真实标签 > 0 的索引 ({len(true_idx)} 个):")
#         print(true_idx.tolist())

#         print(f"\n预测概率 > 0.65 的索引 ({len(pred_idx)} 个):")
#         print(pred_idx.tolist())

#         # 7. 可选：打印对应的概率值
#         print(f"\n预测索引对应的概率值:")
#         for idx in pred_idx.tolist():
#             print(f"  class {idx}: prob = {prob[0, idx].item():.4f}")

#     else:
#         doaModel.eval()
#         for ep in range(epoch):





def train_one_epoch(epoch):
    doaModel.train()
    train_loss = 0

    for feas,labels in trainLd:
        feas = feas.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = doaModel(feas)
        loss = criterion(logits, labels)

        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        train_loss += loss.item() * len(feas)
    train_loss = train_loss / len(trainLd.dataset)

    return train_loss

def val_one_epoch(epoch):
    doaModel.eval()
    eval_loss = 0
    with torch.no_grad():
        for feas,labels in testLd:
            feas = feas.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = doaModel(feas)

            loss = criterion(logits,labels)

            eval_loss += loss.item() * len(feas)
        eval_loss = eval_loss / len(testLd.dataset)
    return eval_loss










import os
import matplotlib.pyplot as plt
import pandas as pd
import torch

if __name__ == '__main__':
    train_loss_list = []
    val_loss_list = []

    # 输出目录
    out_dir = '/kaggle/working/outputs'
    os.makedirs(out_dir, exist_ok=True)

    # 训练循环
    for ep in range(epochs):
        train_loss = train_one_epoch(ep)
        eval_loss = val_one_epoch(ep)
        train_loss_list.append(train_loss)
        val_loss_list.append(eval_loss)
        print(f'Epoch: {ep+1} Training Loss: {train_loss:.6f}', flush=True)
        print(f'Epoch: {ep+1} Val Loss: {eval_loss:.6f}', flush=True)

    # 1) 保存模型
    torch.save(doaModel, os.path.join(out_dir, 'doa_model.pth'))

    # 2) 用 pandas 保存 loss 到 csv
    df = pd.DataFrame({
        'epoch': range(1, len(train_loss_list) + 1),
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
    })
    csv_path = os.path.join(out_dir, 'loss.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'saved: {csv_path}')
    print(df)

    # 3) 画图并保存图片
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='train')
    plt.plot(df['epoch'], df['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)

    fig_path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {fig_path}')