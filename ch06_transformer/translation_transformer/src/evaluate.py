import torch
from tqdm import tqdm
from config import *
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch    # 预测核心逻辑，得到批数据预测概率
from tokenizer import ChineseTokenizer, EnglishTokenizer

from nltk.translate.bleu_score import corpus_bleu   # 引入评价指标bleu

# 验证核心逻辑，返回评价指标（准确率）
def evaluate(model, dataloader, tokenizer, device):
    # 用列表记录参考译文和预测译文
    references = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.tolist()  # 转成列表，方便计算

            # 前向传播，得到一批样本的预测结果
            batch_result = predict_batch(model, inputs, tokenizer, device)
            # 合并这一批结果到预测总列表
            predictions.extend( batch_result )
            # 合并这一批的目标值（参考译文）到总列表
            references.extend( [ [target[1:target.index(tokenizer.end_id)]] for target in targets ] )

    # 调库计算bleu评分
    bleu_score = corpus_bleu(references, predictions)
    return bleu_score

# 评估主流程
def run_evaluate():
    # 1. 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取词表
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR / CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR / EN_VOCAB_FILE)
    print("词表加载成功！")

    # 3. 加载模型
    model = TranslationModel( cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id, en_tokenizer.pad_id ).to(device)

    model.load_state_dict( torch.load( MODEL_DIR/BEST_MODEL ) )
    print("模型加载成功！")

    # 4. 获取测试数据集（加载器）
    test_dataloader = get_dataloader(train=False)

    # 5. 调用评估逻辑
    bleu = evaluate(model, test_dataloader, en_tokenizer, device)

    print("评估结果：")
    print("BLEU 评分: ", bleu)

if __name__ == '__main__':
    run_evaluate()