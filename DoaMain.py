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

batch_size = 2048
epochs = 50

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



def TestMultiple(indices):
    """
    同时测试并绘制多个样本
    indices: 列表，例如 [0, 1, 2, 3, 4, 5, 6, 7]
    """
    # 1. 创建 2x4 的子图画布，尺寸大一些避免拥挤
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten() # 将 2x4 的二维数组展平为 1维 (长度 8)，方便索引
    
    doaModel.eval()
    
    # 2. 遍历传入的 8 个索引
    for i, idx in enumerate(indices):
        if i >= len(axes):  # 防止索引超过 8 个
            break
            
        ax = axes[i]  # 获取当前子图区域
        
        # 取测试集样本
        feas, label = test_dataset[idx]
        feas = feas.unsqueeze(0).to(device) # (1, 2, 8, 8)

        # 前向 + Sigmoid
        with torch.no_grad():
            logits = doaModel(feas)
            prob = torch.sigmoid(logits) # (1, 181)

        # 真实标签与预测标签索引
        true_idx = torch.nonzero(label > 0).flatten()
        pred_idx = torch.where(prob[0] > 0.65)[0]

        # 转换为 numpy 用于绘图
        prob_np = prob[0].detach().cpu().numpy()
        true_idx_np = true_idx.cpu().numpy()
        pred_idx_np = pred_idx.cpu().numpy()

        # ================= 开始绘制第 i 张图 =================
        # 画概率曲线
        ax.plot(angleGird, prob_np, color='blue', label='Pred Prob')

        # 画阈值线
        ax.axhline(y=0.65, color='gray', linestyle='--', label='Threshold')

        # 真实标签垂直线
        if len(true_idx_np) > 0:
            ax.vlines(
                x=angleGird[true_idx_np], ymin=0, ymax=1,
                colors='red', linestyles='dashed', linewidth=1.5, label='True Labels'
            )

        # 预测点散点
        if len(pred_idx_np) > 0:
            ax.scatter(
                angleGird[pred_idx_np], prob_np[pred_idx_np],
                color='green', zorder=5, label='Pred > 0.65'
            )

        # 标注真实标签角度值（红色）
        for t_idx in true_idx_np:
            angle_val = angleGird[t_idx]
            ax.text(
                angle_val, 1.02, f"{angle_val:.1f}°",
                color='red', ha='center', fontsize=8, fontweight='bold'
            )

        # 标注预测点角度值（绿色）
        for p_idx in pred_idx_np:
            angle_val = angleGird[p_idx]
            prob_val = prob_np[p_idx]
            ax.text(
                angle_val, prob_val + 0.03, f"{angle_val:.1f}°",
                color='green', ha='center', fontsize=8
            )

        # 设置坐标轴范围和标题
        ax.set_ylim(0, 1.18) # 给顶部的文字留空间
        ax.set_title(f"Sample Index: {idx}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Angle (degree)')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
        
        # 只在第一张图显示图例，避免遮挡其他图
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)
        
        # 简要打印信息（可选）
        print(f"Idx {idx}: True angles {angleGird[true_idx_np]}, Pred angles {angleGird[pred_idx_np]}")

    # ================= 3. 保存和显示 =================
    plt.tight_layout() # 自动调整子图间距
    
    # ⚠️ 核心修正：必须先 savefig 再 show，否则保存出来的是一张白图！
    img_path = '/kaggle/working/prj_doa/doa_music_2x4_Test.png'
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 2x4 结果图已保存至: {img_path}")
    
    plt.show() # 在 Kaggle Cell 中展示图片






if __name__ == '__main__':
    train_loss_list =[]
    val_loss_list = []
    for ep in range(epochs):
        train_loss = train_one_epoch(ep)
        eval_loss = val_one_epoch(ep)
        train_loss_list.append(train_loss)
        val_loss_list.append(eval_loss)
        print(f'Epoch: {ep+1} Training Loss: {train_loss:.6f}', flush=True)
        print(f'Epoch:{ep+1} Val Loss:{eval_loss:.6f}',flush=True)

    plt.figure()
    plt.plot(train_loss_list,label = 'train')
    plt.plot(val_loss_list,label = 'val')
    plt.show()
    TestMultiple([0, 1, 2, 3, 4, 5, 6, 7])