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

# 2. 文件
RAW_DATA_FILE = 'online_shopping_10_cats.csv'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'
VOCAB_FILE = 'vocab.txt'    # 词表文件
BEST_MODEL = 'best_model.pt'    # 最优模型参数文件

# 3. 特殊token
UNK_TOKEN = '<unk>'     # 未登录词
PAD_TOKEN = '<pad>'     # 填充词

# 4. 超参数
SEQ_LEN = 128     # 序列长度
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

LEARNING_RATE = 1e-3
EPOCHS = 20