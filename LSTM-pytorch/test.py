import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from dataset_model import TimeSeriesDataset, LSTMModel

# 读取原始训练数据
train_df = pd.read_csv('D:/PyCharm/untitled7/LSTM_PyTorch_Electric-Load-Forecasting/LSTM-pytorch/ETTh1.csv')

# 使用原始训练数据来拟合MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df[['OT']])

# 读取测试集的数据（假设数据已经是归一化状态）
test_df = pd.read_csv('dataset/test_dataset.csv')

# 将测试数据转换为PyTorch的Tensor
test_data = torch.tensor(test_df['OT'].values, dtype=torch.float32).unsqueeze(1)

# 创建测试集的数据集对象
seq_length = 10
test_dataset = TimeSeriesDataset(test_data, seq_length)

# 创建数据加载器
batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型参数
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载已训练好的模型参数
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()

# 在测试集上进行预测并记录前16轮和前96轮的预测值和真实值
predictions = []
first_16_predictions = []
first_96_predictions = []
true_values_list = []

with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)

        # 记录预测值
        predictions.append(outputs.detach().cpu().numpy())
        true_values_list.append(targets.detach().cpu().numpy())

        # 收集前16组和前96组的数据
        if len(first_16_predictions) < 16:
            first_16_predictions.append(outputs.detach().cpu().numpy())
        if len(first_96_predictions) < 96:
            first_96_predictions.append(outputs.detach().cpu().numpy())

# 将预测结果转换为一维数组
predictions = np.concatenate(predictions).flatten()

# 逆归一化预测结果
predictions_inversed = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# 真实值（假设测试集中的'true_values'也是归一化状态）
true_values = np.concatenate([t for t in true_values_list]).flatten()

# 逆归一化真实值
true_values_inversed = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()


# 定义计算MAPE的函数
def calculate_mape(y_true, y_pred, epsilon=1e-8):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true + epsilon != 0)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100, mask.sum()
# 定义计算MSE、RMSE、MAE、MAPE的函数
def calculate_mse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2), mask.sum()

def calculate_rmse(mse):
    return np.sqrt(mse) if mse > 0 else float('nan')
mse, mse_count = calculate_mse(true_values_inversed, predictions_inversed)
rmse = calculate_rmse(mse)
print(f"RMSE: {rmse}")
# 计算整体MAPE
mape, mape_count = calculate_mape(true_values_inversed, predictions_inversed)

# 输出整体MAPE
print(f"Overall MAPE: {mape}%")

# 计算前16组的MAPE
first_16_predictions = np.concatenate(first_16_predictions).flatten()[:16]
first_16_true_values = true_values[:16]  # 取前16个真实值
first_16_predictions_inversed = scaler.inverse_transform(first_16_predictions.reshape(-1, 1)).flatten()
first_16_true_values_inversed = scaler.inverse_transform(first_16_true_values.reshape(-1, 1)).flatten()
first_16_mape, first_16_mape_count = calculate_mape(first_16_true_values_inversed, first_16_predictions_inversed)
print(f"MAPE for the first 16 groups: {first_16_mape}%")

# 计算前96组的MAPE
first_96_predictions = np.concatenate(first_96_predictions).flatten()[:96]
first_96_true_values = true_values[:96]  # 取前96个真实值
first_96_predictions_inversed = scaler.inverse_transform(first_96_predictions.reshape(-1, 1)).flatten()
first_96_true_values_inversed = scaler.inverse_transform(first_96_true_values.reshape(-1, 1)).flatten()
first_96_mape, first_96_mape_count = calculate_mape(first_96_true_values_inversed, first_96_predictions_inversed)
print(f"MAPE for the first 96 groups: {first_96_mape}%")

# 输出反归一化的预测值和真实值的前20组
print("反归一化的预测值的前20组:", predictions_inversed[:20])
print("反归一化的真实值的前20组:", true_values_inversed[:20])

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(test_df.index[seq_length:], true_values_inversed, label='True Values')
plt.plot(test_df.index[seq_length:], predictions_inversed, label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Load [MW]')
plt.title('Predicted vs True Values')
plt.legend()
plt.show()