from pathlib import Path    # 路径定义

# 1. 目录路径
# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
# 数据目录
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
# 模型目录
MODEL_DIR = ROOT_DIR / 'models'
# 日志目录
LOG_DIR = ROOT_DIR / 'logs'
# 预训练目录
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'

# 2. 文件
RAW_DATA_FILE = 'online_shopping_10_cats.csv'
BEST_MODEL = 'best_model.pt'    # 最优模型参数文件
BERT_MODEL = 'bert-base-chinese'    # BERT模型名称

# 3. 超参数
SEQ_LEN = 128     # 序列长度
BATCH_SIZE = 16
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 768

LEARNING_RATE = 1e-5
EPOCHS = 10