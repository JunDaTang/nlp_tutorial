# 数据预处理
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split    # 划分数据集

from config import *
from tokenizer import JiebaTokenizer


def preprocess():
    print("-------开始数据预处理...-------")

    # 1. 读取csv文件，得到DataFrame；并提取两列，去除缺失值
    df = pd.read_csv(RAW_DATA_DIR / RAW_DATA_FILE, usecols=['label', 'review'], encoding='utf-8').dropna()

    # 3. 对原始语料做划分，按label分层抽样
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    # 4. 分词并构建词表、保存到文件
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), MODEL_DIR/VOCAB_FILE)

    # 5. 创建分词器
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)

    # 6. 构建数据集
    train_df['review'] = train_df['review'].apply( lambda review: tokenizer.encode(review, SEQ_LEN) )
    test_df['review'] = test_df['review'].apply( lambda review: tokenizer.encode(review, SEQ_LEN) )

    # 7. 保存数据集到文件
    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print("-------数据预处理完成-------")


if __name__ == '__main__':
    preprocess()