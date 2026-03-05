import torch
from tqdm import tqdm
from config import *
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_topk    # 预测核心逻辑，得到topk的id列表
from tokenizer import JiebaTokenizer

# 验证核心逻辑，返回评价指标（top1和top5准确率）
def evaluate(model, dataloader, device):
    top1_acc_count, top5_acc_count = 0, 0
    total_count = 0

    with torch.no_grad():
        # 按批次前向传播
        for inputs, targets in tqdm(dataloader, desc='评估：'):
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播，得到预测5个id
            top5_indices_list = predict_topk(model, inputs)   # 形状（batch_size, 5）
            # 做拉链，对比预测id列表和目标id
            for target, top5_indices in zip(targets, top5_indices_list):
                total_count += 1
                # 判断预测的第一个id是否是target
                if target == top5_indices[0]:
                    top1_acc_count += 1
                # 判断target是否在预测列表中
                if target in top5_indices:
                    top5_acc_count += 1
    top1_acc = top1_acc_count / total_count
    top5_acc = top5_acc_count / total_count
    return top1_acc, top5_acc

# 评估主流程
def run_evaluate():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取词表
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)
    print("词表加载成功！")

    # 3. 加载模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict( torch.load( MODEL_DIR/BEST_MODEL ) )
    print("模型加载成功！")

    # 4. 获取测试数据集（加载器）
    test_dataloader = get_dataloader(train=False)

    # 5. 调用评估逻辑
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)

    print("评估结果：")
    print("top1_acc: ", top1_acc)
    print("top5_acc: ", top5_acc)

if __name__ == '__main__':
    run_evaluate()