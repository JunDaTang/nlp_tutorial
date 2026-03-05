import torch
from torch import nn, optim

from tqdm import tqdm   # 进度条工具

from config import *
from dataset import get_dataloader  # 获取数据加载器
from model import TranslationModel  # 模型

from torch.utils.tensorboard import SummaryWriter   # 日志写入器
import time # 时间库

from tokenizer import ChineseTokenizer, EnglishTokenizer    # 分词器

# 定义训练引擎函数：训练一个epoch，返回平均损失
def train_one_epoch(model, train_loader, loss, optimizer, device):
    model.train()

    total_loss = 0

    # 按批次进行迭代
    for inputs, targets in tqdm(train_loader, desc='训练：'):
        inputs, targets = inputs.to(device), targets.to(device)    # 形状 (N=64, L)
        # 0. 准备参数
        # 0.1 基于目标序列，得到解码的输入和目标 (N, T=tgt_len)
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        # 0.2 源序列填充掩码，(N, S)
        src_pad_mask = (inputs == model.cn_embedding.padding_idx)
        # 0.3 目标序列自注意力掩码 (T, T)
        tgt_mask = model.transformer.generate_square_subsequent_mask( decoder_inputs.shape[1] )

        # 1. 前向传播，(N, T, en_vocab_size)
        decoder_outputs = model(src=inputs, tgt=decoder_inputs, src_pad_mask=src_pad_mask, tgt_mask=tgt_mask)

        # 2. 计算损失，输出形状 (N, vocab_size, L)，目标形状 (N, L)
        loss_value = loss(decoder_outputs.transpose(1, 2), decoder_targets)

        # 3. 反向传播
        loss_value.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 梯度清零
        optimizer.zero_grad()
        # 累加损失
        total_loss += loss_value.item()
    return total_loss / len(train_loader)

# 训练整体流程
def train():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 创建数据加载器
    train_loader = get_dataloader()

    # 3. 获取词表，创建分词器
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)

    # 4. 定义模型
    model = TranslationModel( cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id, en_tokenizer.pad_id ).to(device)

    # 5. 定义损失函数
    loss = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)

    # 6. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 定义一个tensorboard写入器
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 8. 核心训练流程，按epoch进行迭代
    min_loss = float('inf')     # 记录最小训练损失
    for epoch in range(EPOCHS):
        print("="*10, f"EPOCH:{epoch+1}", "="*10)
        this_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        print("本轮训练损失:", this_loss)

        # 将损失写入日志
        writer.add_scalar('loss', this_loss, epoch+1)

        # 判断损失是否下降，保存最优模型
        if this_loss < min_loss:
            min_loss = this_loss
            torch.save( model.state_dict(), MODEL_DIR / BEST_MODEL )
            print("模型保存成功！")
    # 关闭写入器
    writer.close()

if __name__ == '__main__':
    train()