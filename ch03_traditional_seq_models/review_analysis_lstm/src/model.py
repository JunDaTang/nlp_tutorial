import torch
import torch.nn as nn

from config import *

# 自定义神经网络类（基于LSTM）
class ReviewAnalysisModel(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层，指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向LSTM
        self.lstm = nn.LSTM(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # 全连接层，预测输出维度为1（代表属于正向评价的评分）
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=1)

    # 前向传播
    def forward(self, x):
        # 词嵌入
        embed = self.embedding(x)
        # LSTM前向传播，得到输出 (N, L, hidden_size)
        output, (_, _) = self.lstm(embed)
        # 用列表索引取每条数据真实的最后一个隐状态，作为上下文特征向量
        indices = torch.arange( output.shape[0] )
        lengths = ( x != self.embedding.padding_idx ).sum(dim=1)    # 计算每个序列的"真实长度"
        features = output[indices, lengths - 1]

        # 全连接层整合特征，得到预测输出，形状 (N, 1)，再降维成一维 (N,)
        result = self.linear(features).squeeze(-1)

        return result

if __name__ == '__main__':
    vocab_size = 1000
    # 定义测试数据
    input = torch.randint(vocab_size, size=(64, 5))
    # 定义模型
    model = ReviewAnalysisModel(vocab_size, padding_idx=0)
    # 前向传播
    output = model(input)
    print(output.shape)