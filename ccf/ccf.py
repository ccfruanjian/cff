import numpy as np
import pandas as pd
import sklearn.datasets as sd
import sklearn.model_selection as sms
import matplotlib.pyplot as plt
import math
import random
import time
from openpyxl import load_workbook
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

edge_path = 'edge_test_4_A.csv'
node_path = 'node_test_4_A.csv'
edge_data = pd.read_csv(edge_path)
node_data = pd.read_csv(node_path)
node_mean =node_data.iloc[:, 2:].mean()
edge_mean =edge_data.iloc[:, 2:4].mean()
node_stds =node_data.iloc[:, 2:].std()
edge_stds =edge_data.iloc[:, 2:4].std()
node_data.iloc[:, 2:] = node_data.iloc[:, 2:].fillna(node_mean)
edge_data.iloc[:, 2:] = edge_data.iloc[:, 2:].fillna(edge_mean)
edge_outliers = (edge_data.iloc[:, 2:4] > edge_mean + 3 * edge_stds) | (edge_data.iloc[:, 2:4] < edge_mean - 3 * edge_stds)
node_outliers = (node_data.iloc[:, 2:] > node_mean + 3 * node_stds) | (node_data.iloc[:, 2:] < node_mean - 3 * node_stds)
edge_rows_to_drop = edge_outliers.any(axis=1)
node_rows_to_drop = node_outliers.any(axis=1)
node_data = node_data.drop(node_data[node_rows_to_drop].index)
edge_data = edge_data.drop(edge_data[edge_rows_to_drop].index)

1111
21