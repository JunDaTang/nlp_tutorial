import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import *

# 自定义数据集类
class ReviewAnalysisDataset(Dataset):
    # 初始化
    def __init__(self, path):
        # 定义属性，保存所有数据的字典列表
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    # 获取长度
    def __len__(self):
        return len(self.data)

    # 根据index获取元素
    def __getitem__(self, index):
        input = torch.tensor(self.data[index]['review'], dtype=torch.long)
        target = torch.tensor(self.data[index]['label'], dtype=torch.float)
        return input, target

# 获取DataLoader的函数
def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    dataset = ReviewAnalysisDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break