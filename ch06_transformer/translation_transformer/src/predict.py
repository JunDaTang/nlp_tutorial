import torch
from config import *
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer

# 核心预测逻辑函数，返回一批数据的预测概率
def predict_batch(model, inputs, tokenizer, device):
    model.eval()
    # 前向传播
    with torch.no_grad():
        # 定义当前 batch_size (N)
        batch_size = inputs.shape[0]

        # 1. 前向传播
        # 1.1 编码
        src_pad_mask = (inputs == model.cn_embedding.padding_idx)
        memory = model.encode(inputs, src_pad_mask)

        # 1.2 解码：自回归生成
        # 1.2.1 构建第一时间步的输入，长度为 N 的向量 (N, T=1)，内容全部为 <sos> 的 id
        decoder_input = torch.full(size=(batch_size, 1), fill_value=tokenizer.start_id).to(device)

        # 保存生成的id列表
        generated_ids = []
        # 定义一个长度为 N 的 tensor，保存每个样本是否已生成<eos>
        is_finished = torch.full(size=[batch_size], fill_value=False).to(device)

        # 1.2.2 循环迭代，自回归生成
        for i in range(SEQ_LEN):
            # (1) 解码，decoder_output 形状 (N, T, en_vocab_size)
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1])
            decoder_output = model.decode(decoder_input, memory, tgt_mask=tgt_mask, memory_pad_mask=src_pad_mask)
            # (2) 词选择：贪心解码，得到预测下一个词的id (N, L=1)
            next_token_ids = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)
            # (3) 保存预测id到生成列表中
            generated_ids.append(next_token_ids)
            # (4) 更新输入，形状增加 (N, T+1)
            decoder_input = torch.cat((decoder_input, next_token_ids), dim=-1)
            # (5) 判断是否生成 <eos>，如果一批全部生成<eos>则退出循环
            is_finished |= (next_token_ids.squeeze(1) == tokenizer.end_id)
            if is_finished.all():
                break
    # 处理生成结果
    # 基于生成列表 generated_ids: [ tensor(N, 1), tensor(N, 1), ...]
    # (1) 将列表转成 (N, L) 的张量
    generated_tensor =  torch.cat( generated_ids, dim=1 )
    # (2) 转换为二维列表
    generated_list = generated_tensor.tolist()
    # 形如：[ [*, *, *, eos], [*, *, eos, *], [*, *, *, *], ... ]
    # (3) 去掉每个元素（句子的id列表）中，eos之后的所有内容
    for i, sentence_ids in enumerate(generated_list):
        if tokenizer.end_id in sentence_ids:
            eos_pos = sentence_ids.index(tokenizer.end_id)
            generated_list[i] = sentence_ids[:eos_pos]
    # 形如：[ [*, *, *], [*, *], [*, *, *, *], ... ]

    return generated_list # 二维列表返回

def predict(text, model, cn_tokenizer, en_tokenizer, device):
    # 1. 准备数据：文本处理
    # 1.1/1.2 分词、id化
    ids = cn_tokenizer.encode(text)
    # 1.3 转换为 tensor，作为输入
    input = torch.tensor([ids], dtype=torch.long).to(device)

    # 2. 预测
    # 前向传播，得到预测概率
    result = predict_batch(model, input, en_tokenizer, device)

    return en_tokenizer.decode( result[0] )    # 只有唯一的一个数据，解码成英文句子

def run_predict():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取词表，得到分词器
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR / CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR / EN_VOCAB_FILE)
    print("词表加载成功！")

    # 3. 加载模型
    model = TranslationModel( cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id, en_tokenizer.pad_id ).to(device)

    model.load_state_dict( torch.load( MODEL_DIR/BEST_MODEL ) )
    print("模型加载成功！")

    # 6. 程序运行流程
    print("欢迎使用中英翻译模型！输入q或者quit退出...")
    while True: # 核心：一个死循环
        # 捕获用户输入
        user_input = input("中文 > ")
        # 判断：如果是q或者quit，直接退出
        if user_input.strip() in ['q', 'quit']:
            print("欢迎下次再来！")
            break
        # 判断：如果是空白，提示信息后继续循环
        if user_input.strip() == '':
            print("请输入有效内容！")
            continue

        # 预测译文
        result = predict(user_input, model, cn_tokenizer, en_tokenizer, device)
        print("英文：", result)

if __name__ == '__main__':
    # text = "我们公司"
    # top5_tokens = predict(text)
    # print(top5_tokens)
    run_predict()