import torch
from tqdm import tqdm
from config import *
from model import ReviewAnalysisModel
from dataset import get_dataloader
from predict import predict_batch    # 预测核心逻辑，得到批数据预测概率

# 验证核心逻辑，返回评价指标（准确率）
def evaluate(model, dataloader, device):
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        # 按批次前向传播
        for batch in tqdm(dataloader, desc='评估：'):
            labels = batch.pop('labels').tolist()
            inputs = {k: v.to(device) for k, v in batch.items()}

            # 前向传播，得到预测概率
            batch_results = predict_batch(model, inputs)   # 形状（batch_size,）
            # 做拉链，对比每条数据的预测概率和目标分类
            for target, result in zip(labels, batch_results):
                total_count += 1
                # 判断预测概率是否大于0.5，转成预测分类标签
                result = 1 if result > 0.5 else 0
                if result == target:
                    correct_count += 1

    return correct_count / total_count

# 评估主流程
def run_evaluate():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 加载模型
    model = ReviewAnalysisModel().to(device)
    model.load_state_dict( torch.load( MODEL_DIR/BEST_MODEL ) )
    print("模型加载成功！")

    # 4. 获取测试数据集（加载器）
    test_dataloader = get_dataloader(train=False)

    # 5. 调用评估逻辑
    acc = evaluate(model, test_dataloader, device)

    print("评估结果：")
    print("准确率: ", acc)

if __name__ == '__main__':
    run_evaluate()