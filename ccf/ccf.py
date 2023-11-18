import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch import nn
from GAT_LSTM import GATLSTM
import torch

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
for col in edge_data.columns[2:4]:
    col_mean = edge_data[col].mean()
    col_std = edge_data[col].std()
    lower_bound = col_mean - 6 * col_std
    upper_bound = col_mean + 6 * col_std
    edge_data[col] = np.clip(edge_data[col], lower_bound, upper_bound)

# 按照日期分组
grouped_node_data = node_data.groupby('date_id')
# 对id进行编码
le = preprocessing.LabelEncoder()
node = []
label = []
for _, node_data in grouped_node_data:
    node_data.iloc[:, 0] = le.fit_transform(node_data.iloc[:, 0])
    node_data = node_data.sort_values(by='geohash_id')
    node.append(node_data.iloc[:, [0] + list(range(2, 37))])
    y = node_data.iloc[:, [-2, -1]].values
    y = torch.tensor(y.reshape((1, -1))).float()
    # print(y.shape)torch.Size([1, 2280])
    label.append(y)
node_id = list(range(1140))
node_id = le.inverse_transform(node_id)
# print(label.shape)
grouped_edge_data = edge_data.groupby('date_id')
edge = []
for _, edge_data in grouped_edge_data:
    edge_data.iloc[:, 0] = le.fit_transform(edge_data.iloc[:, 0])
    edge_data.iloc[:, 1] = le.fit_transform(edge_data.iloc[:, 1])
    edge.append(edge_data.iloc[:, 0:4])
# print(edge[0].shape)(11930, 4)
nodes = []
edges_index = []
edges_attr = []
for i in range(len(node)):
    x = node[i].iloc[:, 1:].values
    edges = edge[i]
    # 边的头结点
    edge_index1 = edges[['geohash6_point1']].values.astype(int)
    # 边的尾结点
    edge_index2 = edges[['geohash6_point2']].values.astype(int)
    edge_attr = edges[['F_1', 'F_2']].values
    edge_index = np.array([edge_index1, edge_index2]).reshape((2, -1))
    # print(x.shape)(1140, 35)
    # print(edge_index.shape)(2, 11930)
    # print(edge_attr.shape)(11930, 2)
    nodes.append(x)
    edges_index.append(edge_index)
    edges_attr.append(edge_attr)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_size = 35
out_size = 100
h_size = 256
num_heads = 8
edge_dim = 2
window_size = 7
batch_size = 1


def create_sliding_windows(nodes, edges_index, edges_attr, label, window_size):
    node_, edges_index_, edges_attr_, label_ = [], [], [], []
    for i in range(len(nodes) - window_size - 4):
        node_.append(nodes[i:i + window_size])
        edges_index_.append(edges_index[i:i + window_size])
        edges_attr_.append(edges_attr[i:i + window_size])
        label_.append(label[i + window_size:i + window_size + 4])
    return node_, edges_index_, edges_attr_, label_


nodes_, edges_index_, edges_attr_, label_ = create_sliding_windows(nodes, edges_index, edges_attr, label, window_size)
model = GATLSTM(input_size=in_size, hidden_size=64, num_heads=8, num_layers=2, window_size=window_size, device=device)
model.to(device)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
criterion = nn.MSELoss()
num_epochs = 100

model.train()
for epoch in range(num_epochs):
    for i in range(len(nodes_)):
        node = nodes_[i]
        edge_index = edges_index_[i]
        edges_attr = edges_attr_[i]
        label = label_[i]
        label = torch.stack(label, dim=1).to(device)
        # print(label.shape)torch.Size([1, 4, 2280])
        out = model(node, edge_index, edges_attr)
        # print(out.shape)torch.Size([1, 4, 2280])
        # 计算损失
        loss = criterion(out, label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
node_ = nodes[-window_size:]
edges_index_ = edges_index[-window_size:]
edges_attr_ = edges_attr[-window_size:]
y = model(node_, edges_index_, edges_attr_)
y = y.view(-1).tolist()
# 创建一个包含node_data的DataFrame
day1_values = y[:2280]
day2_values = y[2280: 2280 * 2]
day3_values = y[2280 * 2: 2280 * 3]
day4_values = y[2280 * 3:]
# 创建一个包含适当索引的DataFrame

df = pd.DataFrame()
df['node_id'] = node_id
df['20230404']= [(day1_values[i],day1_values[i+1]) for i in range(0, len(day1_values), 2)]
df['20230405']=[(day2_values[i],day2_values[i+1]) for i in range(0, len(day2_values), 2)]
df['20230406']=[(day3_values[i],day3_values[i+1]) for i in range(0, len(day3_values), 2)]
df['20230407']=[(day4_values[i],day4_values[i+1]) for i in range(0, len(day4_values), 2)]

combined_data = []
for index, row in df.iterrows():
    data_0404 = f"\t{row['20230404'][0]}\t{row['20230404'][1]}\t20230404"
    data_0405 = f"\t{row['20230405'][0]}\t{row['20230405'][1]}\t20230405"
    data_0406 = f"\t{row['20230406'][0]}\t{row['20230406'][1]}\t20230406"
    data_0407 = f"\t{row['20230407'][0]}\t{row['20230407'][1]}\t20230407"

    combined_data.append(row['node_id'] + data_0404)
    combined_data.append(row['node_id'] + data_0405)
    combined_data.append(row['node_id'] + data_0406)
    combined_data.append(row['node_id'] + data_0407)
# 创建一个DataFrame
df_combined = pd.DataFrame(combined_data, columns=['geohash_id	consumption_level	activity_level	date_id'])

# 保存为CSV文件
df_combined.to_csv('submit.csv', index=False)
