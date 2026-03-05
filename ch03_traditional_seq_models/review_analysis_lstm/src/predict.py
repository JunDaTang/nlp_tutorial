import torch
from config import *
from model import ReviewAnalysisModel
from tokenizer import JiebaTokenizer

# 核心预测逻辑函数，返回一批数据的预测概率
def predict_batch(model, inputs):
    model.eval()
    # 前向传播
    with torch.no_grad():
        outputs = model(inputs)   # 形状（batch_size,）
    # 转换为预测概率
    batch_results = torch.sigmoid(outputs)
    return batch_results.tolist() # 转换成列表返回

def predict(text, model, tokenizer, device):
    # 1. 准备数据：文本处理
    # 1.1/1.2 分词、id化
    ids = tokenizer.encode(text, SEQ_LEN)
    # 1.3 转换为 tensor，作为输入
    input = torch.tensor([ids], dtype=torch.long).to(device)

    # 2. 预测
    # 前向传播，得到预测概率
    result = predict_batch(model, input)

    return result[0]    # 只有唯一的一个数据

def run_predict():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取词表
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print("词表加载成功！")

    # 3. 加载模型
    model = ReviewAnalysisModel(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_id).to(device)
    model.load_state_dict( torch.load( MODEL_DIR/BEST_MODEL ) )
    print("模型加载成功！")

    # 6. 程序运行流程
    print("欢迎使用文本情感分析模型！输入q或者quit退出...")
    while True: # 核心：一个死循环
        # 捕获用户输入
        user_input = input("> ")
        # 判断：如果是q或者quit，直接退出
        if user_input.strip() in ['q', 'quit']:
            print("欢迎下次再来！")
            break
        # 判断：如果是空白，提示信息后继续循环
        if user_input.strip() == '':
            print("请输入有效内容！")
            continue

        # 根据预测概率，判断是正向还是负向评价
        result = predict(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f"正向评价 (置信度: {result})")
        else:
            print(f"负向评价 (置信度：{1 - result})")

if __name__ == '__main__':
    # text = "我们公司"
    # top5_tokens = predict(text)
    # print(top5_tokens)
    run_predict()