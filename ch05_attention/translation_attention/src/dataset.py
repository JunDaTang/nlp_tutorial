import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import *

from torch.nn.utils.rnn import pad_sequence     # 序列填充

# 自定义数据集类
class TranslationDataset(Dataset):
    # 初始化
    def __init__(self, path):
        # 定义属性，保存所有数据的字典列表
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    # 获取长度
    def __len__(self):
        return len(self.data)

    # 根据index获取元素
    def __getitem__(self, index):
        input = torch.tensor(self.data[index]['cn'], dtype=torch.long)
        target = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input, target

# 定义一个整理函数，将一批数据长度对齐（填充）
def collate_fn(batch):
    # batch形如 [ (input0, target0), (input1, target1)，...]；先分成inputs和targets两个列表
    input_tensor_list = [ item[0] for item in batch ]
    target_tensor_list = [ item[1] for item in batch ]
    # 合并成长度对齐的一个batch tensor
    input_batch_tensor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
    target_batch_tensor = pad_sequence(target_tensor_list, batch_first=True, padding_value=0)

    return input_batch_tensor, target_batch_tensor

# 获取DataLoader的函数
def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    dataset = TranslationDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    # for input, target in train_dataloader:
    #     print(input.shape, target.shape)
    #     break

    data_iter = iter(train_dataloader)
    input_batch, target_batch = next(data_iter)
    print(input_batch.shape)
    print(target_batch.shape)

    input_batch, target_batch = next(data_iter)
    print(input_batch.shape)
    print(target_batch.shape)