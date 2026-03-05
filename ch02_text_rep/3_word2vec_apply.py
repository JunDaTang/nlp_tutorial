import torch
import torch.nn as nn

from gensim.models import KeyedVectors  # 加载词向量

import jieba

# 1. 加载预训练的词向量
wv_model = KeyedVectors.load_word2vec_format('data/word2vec.kv')

# 获取词向量维度
vector_dim = wv_model.vector_size
print("词向量维度：", vector_dim)

# 增加对 OOV 问题的处理
unk_token = '<unk>'
id2word = [unk_token] + wv_model.index_to_key
word2id = { word : id for id, word in enumerate(id2word) }

# 2. 构建词表，获取词表大小
# word2id = wv_model.key_to_index
vocab_size = len(word2id)
print("词表大小：", vocab_size)

# 3. 构造一个词向量矩阵 tensor
embedding_matrix = torch.zeros(vocab_size, vector_dim)
for word, id in word2id.items():
    if word in wv_model:
        embedding_matrix[id] = torch.tensor(wv_model[word])

print(embedding_matrix.shape)

# 4. 构建神经网络的嵌入层
embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
# embedding = nn.Embedding.from_pretrained( torch.tensor(wv_model.vectors), freeze=True )

# 5. 测试
text = "我喜欢乘坐宇宙飞船"

# 5.1 分词
tokens = jieba.lcut(text)
print(tokens)

# 5.2 将语料id化
ids = [ word2id.get(word, word2id[unk_token]) for word in tokens ]
print(ids)

# 5.3 构造神经网络输入数据（tensor）
input = torch.tensor([ids])

# 5.4 前向传播：查找词向量
output = embedding(input)

print(output)
print(output.shape)