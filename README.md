# AI 深度學習項目集合

這個資料夾包含多個深度學習項目，主要使用 TensorFlow/Keras 框架進行圖像分類任務。

---

## 📁 項目結構

```
ai/
├── cifar-10.py           # CIFAR-10 圖像分類模型
├── cifar-10.md           # CIFAR-10 項目說明文檔
├── MNIST.py              # MNIST 手寫數字識別模型
├── README.md             # 本文件
└── outputs/              # 訓練結果輸出文件夾
    ├── best_model.keras           # 最佳模型
    ├── cifar10_cnn_improved.keras # 最終訓練模型
    ├── training_history.csv       # 訓練歷史數據
    ├── model_summary.txt          # 模型架構摘要
    ├── final_result.txt           # 最終評估結果
    └── accuracy_loss_curve.png    # 精度和損失曲線圖

```

---

## 🎯 CIFAR-10 CNN 項目

### 項目概述
使用卷積神經網絡 (CNN) 對 CIFAR-10 數據集進行圖像分類。CIFAR-10 包含 10 個類別（飛機、汽車、鳥類等），共 60,000 張 32×32 的彩色圖像。

### 主要特性

✅ **數據處理**
- 自動加載 CIFAR-10 數據集 (50,000 訓練 + 10,000 測試)
- 分割為 40,000 訓練集和 10,000 驗證集
- 像素值歸一化到 [0, 1] 範圍

✅ **數據增強**
- 隨機水平翻轉 (Horizontal Flip)
- 隨機平移 (Translation): ±10%
- 隨機旋轉 (Rotation): ±5°

✅ **CNN 模型架構**
- 3 個卷積塊，逐漸增加濾波器數量 (64 → 128 → 256)
- 批量歸一化 (Batch Normalization) 加速訓練
- 全局平均池化層 (Global Average Pooling)
- 4 層 Dropout 防止過擬合 (0.25 → 0.30 → 0.40 → 0.50)

✅ **訓練優化**
- 優化器: Adam (學習率: 0.001)
- 損失函數: 稀疏交叉熵 (sparse_categorical_crossentropy)
- 提前停止 (EarlyStopping): patience=8
- 動態學習率調整 (ReduceLROnPlateau): factor=0.5, patience=3
- 模型檢查點保存最佳模型

### 🔧 超參數詳情

| 超參數 | 值 | 說明 |
|--------|-----|------|
| **Epochs** | 40 | 最大訓練輪次 |
| **Batch Size** | 64 | 每個批次的樣本數 |
| **Learning Rate** | 0.001 | Adam 優化器初始學習率 |
| **Conv Filters** | 64, 128, 256 | 卷積層濾波器數 |
| **Kernel Size** | (3, 3) | 卷積核大小 |
| **Dropout Rates** | 0.25, 0.30, 0.40, 0.50 | 各層比例 |
| **Data Aug. - Shift** | 0.1 | 10% 隨機平移 |
| **Data Aug. - Rotation** | 0.05 | 5° 隨機旋轉 |
| **EarlyStopping** | 8 | 驗證損失無改進時等待輪數 |
| **ReduceLROnPlateau** | 0.5 | 學習率衰減因子 |

### 📊 模型架構

```
Input (32, 32, 3)
   ↓
Data Augmentation Layer
   ↓
[Conv Block 1] 64 filters × 2 + BatchNorm + ReLU + MaxPool + Dropout(0.25)
   ↓
[Conv Block 2] 128 filters × 2 + BatchNorm + ReLU + MaxPool + Dropout(0.30)
   ↓
[Conv Block 3] 256 filters + BatchNorm + ReLU + MaxPool + Dropout(0.40)
   ↓
Global Average Pooling 2D
   ↓
Dense (256, ReLU) + Dropout(0.50)
   ↓
Dense (10, Softmax) → Output [10 classes]
```

### 🚀 如何運行

#### 1. 安裝依賴
```bash
pip install tensorflow numpy matplotlib
```

#### 2. 運行程式
```bash
python cifar-10.py
```

#### 3. 屬性信息（程式會自動創建 outputs 資料夾）
- 開始加載 CIFAR-10 數據集
- 構建和編譯 CNN 模型
- 開始訓練 (約，取決於硬件)
- 評估模型性能
- 保存訓練成果

### 📈 輸出文件說明

| 文件名 | 說明 |
|--------|------|
| `best_model.keras` | 驗證損失最小的模型檢查點 |
| `cifar10_cnn_improved.keras` | 最終訓練完成的模型 |
| `training_history.csv` | 每個 epoch 的損失和精度數據 |
| `model_summary.txt` | 完整的模型架構和參數數量 |
| `final_result.txt` | 訓練、驗證、測試集的最終精度和損失 |
| `accuracy_loss_curve.png` | 訓練曲線可視化圖表 |

### 📝 期望結果

訓練完成後，會在 `final_result.txt` 中生成如下結果：
```
Train Accuracy:      0.XXXXXX
Validation Accuracy: 0.XXXXXX
Train Loss:          X.XXXXXX
Validation Loss:     X.XXXXXX
Test Loss:           X.XXXXXX
Test Accuracy:       0.XXXXXX
```

---

## 🔍 MNIST 項目

`MNIST.py` 是手寫數字識別程式，使用相似的 CNN 架構對 MNIST 數據集進行分類。

---

## 🛠️ 技術棧

- **框架**: TensorFlow 2.x, Keras
- **數據處理**: NumPy
- **可視化**: Matplotlib
- **語言**: Python 3.x

---

## 📌 項目流程

```
1. 數據準備
   ├─ 加載 CIFAR-10 數據集
   ├─ 歸一化像素值
   ├─ 打亂數據順序
   └─ 分割訓練/驗證/測試集

2. 模型構建
   ├─ 定義數據增強層
   └─ 堆疊卷積、池化、Dropout 層

3. 模型編譯
   └─ 配置優化器、損失函數、指標

4. 訓練
   ├─ 應用數據增強
   ├─ 進行梯度下降優化
   ├─ 監控驗證性能
   └─ 應用回調函數 (EarlyStopping, ReduceLROnPlateau 等)

5. 評估
   ├─ 計算訓練/驗證/測試精度
   └─ 生成性能報告

6. 可視化
   ├─ 繪製精度曲線
   ├─ 繪製損失曲線
   └─ 保存結果圖表
```

---

## 💡 優化建議

如需提升模型性能，可嘗試：

1. **模型架構調整**
   - 增加卷積層數量或濾波器數
   - 嘗試不同的激活函數 (Leaky ReLU, GELU 等)

2. **超參數調優**
   - 調整 learning rate、batch size、dropout rates
   - 使用學習率衰減或循環學習率

3. **數據增強擴展**
   - 添加更多增強方式 (色彩抖動、縮放等)
   - 嘗試 Mixup 或 CutMix 等進階技術

4. **訓練策略**
   - 實現 K-Fold 交叉驗證
   - 使用 Ensemble 方法組合多個模型

---

## 📝 備註

- 程式設置了固定 seed (10) 確保結果可重現
- 訓練過程會自動監控驗證損失，提前停止以避免過擬合
- 所有結果自動保存到 `outputs/` 資料夾

---

## 📧 其他說明

- 修改 `cifar-10.py` 中的超參數可改變模型性能
- 支持 GPU 加速訓練 (需要安裝 TensorFlow GPU 版本)
- 訓練時間取決於硬件配置

