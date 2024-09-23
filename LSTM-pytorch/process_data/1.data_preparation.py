import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_csv('D:/PyCharm/untitled7/LSTM_PyTorch_Electric-Load-Forecasting/LSTM-pytorch/ETTh1.csv')

# 处理缺失值
#df.fillna(method='ffill', inplace=True)  # 使用前向填充方法填充缺失值

# 去除逗号并将电力负荷值转换为浮点数
#df['OT'] = df['OT'].str.replace(',', '').astype(float)

# 将时间列转换为日期时间格式
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
#df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], format='%d/%m/%Y %H:%M:%S')

# 按升序对数据进行排序
df.sort_values('date', ascending=True, inplace=True)

# 只保留 Elia Grid Load 列
df = df[['OT']]

# 数据归一化
scaler = MinMaxScaler()
df['OT'] = scaler.fit_transform(df[['OT']])

# 保存处理后的数据集
df.to_csv('../dataset/processed_dataset.csv', index=False)
print('Data Preparation Complete')