from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
import DOADataSets, DOAmodel
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import config
import os
import pandas as pd

args = config.get_options()


batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
out_dir = args.out_dir


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
train_dataset, val_dataset = random_split(small_dataset, [train_size, test_size])

# 2. 模型搬到 GPU
doaModel = DOAmodel.DOAModel(angleGird.shape[0], 2).to(device)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU 训练")
    doaModel = nn.DataParallel(doaModel)

# 3. DataLoader 优化（num_workers 和 pin_memory 能加速 GPU 训练）
trainLd = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, 
    num_workers=2, pin_memory=True
)
valLd = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=2, pin_memory=True
)

C = len(angleGird)
pos_weight = torch.full((C,), 11.0)
pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(doaModel.parameters(), lr=lr)



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
        for feas,labels in valLd:
            feas = feas.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = doaModel(feas)

            loss = criterion(logits,labels)

            eval_loss += loss.item() * len(feas)
        eval_loss = eval_loss / len(valLd.dataset)
    return eval_loss




class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float('inf')
        self.no_improve = 0
        self.best_ep = 0

    def step(self, val_loss, ep):
        """每个 epoch 验证后调用。返回 True 表示该停了。"""
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.no_improve = 0
            self.best_ep = ep
        else:
            self.no_improve += 1

        print(f"  [best val {self.best_val:.6f} @ epoch {self.best_ep+1}]  "
              f"无改善 {self.no_improve}/{self.patience}", flush=True)
        return self.no_improve >= self.patience









if __name__ == '__main__':
    train_loss_list = []
    val_loss_list = []

    es = EarlyStopping(patience=20, min_delta=1e-5)   # ← 建一个对象
    # 输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 训练循环
    for ep in range(epochs):
        train_loss = train_one_epoch(ep)
        eval_loss = val_one_epoch(ep)
        train_loss_list.append(train_loss)
        val_loss_list.append(eval_loss)
        print(f'Epoch: {ep+1} Training Loss: {train_loss:.6f}', flush=True)
        print(f'Epoch: {ep+1} Val Loss: {eval_loss:.6f}', flush=True)
        if es.step(eval_loss, ep):                    # ← 判断是否停止
            print(f"=> Early stopping at epoch {ep+1}, best was epoch {es.best_ep+1}", flush=True)
            break

    # 1) 保存模型
    weight_path = os.path.join(out_dir, 'model_weight.pth')
    if torch.cuda.device_count() > 1:
        # 重要：使用 model.module.state_dict() 来去除 DataParallel 添加的 "module." 前缀
        torch.save(doaModel.module.state_dict(), weight_path)
    else:
        torch.save(doaModel.state_dict(), weight_path)

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