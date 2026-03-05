import torch
import torch.nn as nn

from config import *

from transformers import AutoModel

# 自定义神经网络类（基于BERT）
class ReviewAnalysisModel(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # BERT层
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_DIR/BERT_MODEL)
        # 全连接层，预测输出维度为1（代表属于正向评价的评分）
        self.linear = nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)

    # 前向传播
    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT 前向传播，得到输出
        output = self.bert(input_ids, attention_mask, token_type_ids)
        # 提取output中CLS对应的输出隐状态，形状 (N, hidden_size)
        cls_hidden_stat = output.pooler_output

        # 全连接层整合特征，得到预测输出，形状 (N, 1)，再降维成一维 (N,)
        result = self.linear(cls_hidden_stat).squeeze(-1)

        return result

if __name__ == '__main__':

    model = ReviewAnalysisModel()
    print(model)