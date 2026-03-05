import torch
import jieba
from config import *
from model import InputMethodModel
from tokenizer import JiebaTokenizer

# 核心预测逻辑函数，返回 topk 的预测 id
def predict_topk(model, input, k = 5):
    model.eval()
    # 前向传播
    with torch.no_grad():
        output = model(input)   # 形状（batch_size = 1, vocab_size）
    # 取输出的topk的id列表
    top_indices = torch.topk(output, k).indices
    return top_indices.tolist() # 转换成列表返回

def predict(text, model, tokenizer, k, device):
    # 1. 准备数据：文本处理
    # 1.1/1.2 分词、id化
    ids = tokenizer.encode(text)
    # 1.3 转换为 tensor，作为输入
    input = torch.tensor([ids], dtype=torch.long).to(device)

    # 2. 预测
    # 2.1 前向传播，得到 topk 的 id 列表
    top_indices_list = predict_topk(model, input, k=k)
    # 2.2 将id列表转换成token列表
    top_tokens = [tokenizer.id2word[id] for id in top_indices_list[0]]

    return top_tokens

def run_predict():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取词表
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)
    print("词表加载成功！")

    # 3. 加载模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict( torch.load( MODEL_DIR/BEST_MODEL ) )
    print("模型加载成功！")

    # 6. 程序运行流程
    print("欢迎使用智能输入法模型！输入q或者quit退出...")
    input_history = ''  # 保存历史输入序列
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
        # 将当前输入添加到历史输入中，作为input
        input_history += user_input
        # 预测tokens
        top_indices = predict(input_history, model, tokenizer, 5, device)
        print("预测结果：", top_indices)


if __name__ == '__main__':
    # text = "我们公司"
    # top5_tokens = predict(text)
    # print(top5_tokens)
    run_predict()