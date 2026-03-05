import torch
import torch.nn as nn

from config import *


# 自定义编码器（基于GRU）
class TranslationEncoder(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层，指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)

    # 前向传播
    def forward(self, x):
        # 词嵌入
        embed = self.embedding(x)
        # GRU前向传播，得到输出 (N, L, hidden_size)
        output, _ = self.gru(embed)
        # 用列表索引取每条数据真实的最后一个隐状态，作为上下文特征向量
        indices = torch.arange( output.shape[0] )
        lengths = ( x != self.embedding.padding_idx ).sum(dim=1)    # 计算每个序列的"真实长度"
        features = output[indices, lengths - 1]

        return features


# 自定义解码器（基于GRU）
class TranslationDecoder(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层，指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # 全连接层
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    # 前向传播，传入初始隐状态
    def forward(self, x, h0=None):
        # 1. 词嵌入
        embed = self.embedding(x)
        # 2. 通过GRU得到输出隐状态（特征向量）
        output, hn = self.gru(embed, h0)
        # 3. 整合特征，预测分类
        output = self.linear(output)

        return output, hn

# 自定义Seq2Seq模型
class TranslationModel(nn.Module):
    # 初始化
    def __init__(self, cn_vocab_size, en_vocab_size, cn_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(cn_vocab_size, cn_padding_idx)
        self.decoder = TranslationDecoder(en_vocab_size, en_padding_idx)

if __name__ == '__main__':
    # 定义模型
    model = TranslationModel(1000, 1024, 0, 0)
    print(model.encoder)
    print(model.decoder)