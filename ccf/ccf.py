import numpy as np
import pandas as pd
import torch.nn as nn1
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LSTM import LSTM
import sys


import torch
# submit_path = 'submit_example.csv'
# submit_data = pd.read_csv(submit_path)
# submit_data['geohash_id', 'consumption_level', 'activity_level', 'date_id'] = submit_data['geohash_id consumption_level  activity_level date_id'].str.extract(r'(\w+)\s+(\d+)\s+(\d+)\s+(\d+)')
# submit_data.to_csv('new_file.csv', index=False)
times_to_pred = 4
scaler = MinMaxScaler()
edge_path = 'edge_90.csv'
node_path = 'train_90.csv'
edge_data = pd.read_csv(edge_path)
node_data = pd.read_csv(node_path)
node_mean = node_data.iloc[:, 2:].mean()
edge_mean = edge_data.iloc[:, 2:4].mean()
node_stds = node_data.iloc[:, 2:].std()
edge_stds = edge_data.iloc[:, 2:4].std()
node_data.iloc[:, 2:] = node_data.iloc[:, 2:].fillna(node_mean)
edge_data.iloc[:, 2:] = edge_data.iloc[:, 2:].fillna(edge_mean)
for col in node_data.columns[2:]:
    col_mean = node_data[col].mean()
    col_std = node_data[col].std()
    lower_bound = col_mean - 6 * col_std
    upper_bound = col_mean + 6 * col_std
    node_data[col] = np.clip(node_data[col], lower_bound, upper_bound)

# 替换边数据中的异常值
# for col in edge_data.columns[2:4]:
#     col_mean = edge_data[col].mean()
#     col_std = edge_data[col].std()
#     lower_bound = col_mean - 6 * col_std
#     upper_bound = col_mean + 6 * col_std
#     edge_data[col] = np.clip(edge_data[col], lower_bound, upper_bound)

# 按照id分组
grouped_node_data = node_data.groupby('geohash_id')
# 对id进行编码
le = preprocessing.LabelEncoder()
node_data.iloc[:, 0] = le.fit_transform(node_data.iloc[:, 0])

node = []
label=[]
for _, node_data in grouped_node_data:
    node_data=node_data.sort_values(by='geohash_id')
    temp=node_data.iloc[:, [0] + list(range(2, 37))]

    node.append(temp.values)
    y=node_data.iloc[:,[-2,-1]].values
    label.append(y)

node_id= list(range(1140))
node_id=le.inverse_transform(node_id)
# print(label.shape)
# grouped_edge_data = edge_data.groupby('date_id')
# edge_data.iloc[:, 0] = le.fit_transform(edge_data.iloc[:, 0])
# edge_data.iloc[:, 1] = le.fit_transform(edge_data.iloc[:, 1])
# print(edge_data)
# edge = []
# for _, edge_data in grouped_edge_data:
#
#     edge.append(edge_data.iloc[:, 0:4])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_sliding_windows(data,label, window_size):
    x,y=[],[]
    for i in range(len(data) - window_size-1):
        x.append(data[i:i + window_size])
        y.append(label[i + window_size])
    return np.array(x),np.array(y)
def train_model(model, X_train, y_train, num_epochs):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(num_epochs):
        model.hidden = model.reset_hidden_state()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float().to(device), y_train)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

    return model.eval()

# model.to(device)
#设置随机数种子
torch.manual_seed(1234)
#窗口大小
window_size =7
batch_size = 1
#data中的元素表示一个节点的所有信息
res = {}
for data,label in zip(node,label):
    x,y= create_sliding_windows(data, label, window_size)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    input_size = 36
    hidden_size = 150
    num_layers = 2
    output_size = 2
    num_epochs = 100
    LSTM_model = LSTM(batch_size=batch_size, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,device=device)
    LSTM_model.to(device)
    x = x.to(device)  # 将训练数据移动到GPU设备
    y = y.to(device)
    model = train_model(LSTM_model, x , y , num_epochs)
    with torch.no_grad():
        model = model.to(device)
        model.hidden = (model.hidden[0].cpu(), model.hidden[1].cpu())
        test_seq = x[-1:].to(device)
        preds = []
        for time in range(times_to_pred):
            y_test_pred = model(test_seq)
            pred1 = torch.flatten(y_test_pred[0][0]).item()
            pred2 = torch.flatten(y_test_pred[0][1]).item()
            preds.append(pred1)
            preds.append(pred2)
            new_seq = test_seq.cpu().numpy().flatten()
            new_seq = np.append(new_seq, [pred1])
            new_seq = np.append(new_seq, [pred2])
            test_seq = torch.as_tensor(new_seq).view(1, window_size, 36).float()
    #     true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
    predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()
    print(preds, predicted_cases)
    res[data] = predicted_cases



# seq=create_sliding_windows(data,label,window_size)
# # print(X.shape)
# # print(y.shape)


# model.eval()
# x=data[-5:,:]
# x=x.reshape(1,5,-1)
# x=x.to(device)
# y=model(x)
# y=y.view(-1).tolist()
# # 创建一个包含node_data的DataFrame
# day1_values = y[:2280]
# day2_values = y[2280: 2280 * 2]
# day3_values = y[2280 * 2: 2280 * 3]
# day4_values = y[2280 * 3:]
# # 创建一个包含适当索引的DataFrame
#
# df = pd.DataFrame()
# df['node_id']=node_id
# df['20230404']= [(round(day1_values[i], 3), round(day1_values[i+1], 2)) for i in range(0, len(day1_values), 2)]
# df['20230405']=[(round(day2_values[i], 3), round(day2_values[i+1], 2)) for i in range(0, len(day2_values), 2)]
# df['20230406']=[(round(day3_values[i], 3), round(day3_values[i+1], 2)) for i in range(0, len(day3_values), 2)]
# df['20230407']=[(round(day4_values[i], 3), round(day4_values[i+1], 2)) for i in range(0, len(day4_values), 2)]
#
#
# combined_data = []
# for index, row in df.iterrows():
#     data_0404 = f"\t{row['20230404'][0]}\t{row['20230404'][1]}\t20230404"
#     data_0405 = f"\t{row['20230405'][0]}\t{row['20230405'][1]}\t20230405"
#     data_0406 = f"\t{row['20230406'][0]}\t{row['20230406'][1]}\t20230406"
#     data_0407 = f"\t{row['20230407'][0]}\t{row['20230407'][1]}\t20230407"
#
#     combined_data.append(row['node_id']+data_0404)
#     combined_data.append(row['node_id']+data_0405)
#     combined_data.append(row['node_id'] + data_0406)
#     combined_data.append(row['node_id'] + data_0407)
# # 创建一个DataFrame
# df_combined = pd.DataFrame(combined_data, columns=['geohash_id	consumption_level	activity_level	date_id'])
#
# # 保存为CSV文件
# df_combined.to_csv('submit.csv', index=False)


