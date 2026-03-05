# 数据预处理
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split    # 划分数据集

from config import *
from tokenizer import JiebaTokenizer

# 构建数据集的函数，传入原始语料和词表 word2id，返回 {'input':[ids], 'target': id}
def build_dataset(sentences, tokenizer):
    # 1. 将所有句子进行分词、id化
    sentences_id = [ tokenizer.encode(sentence) for sentence in sentences ]

    # 2. 构建input和target组成的dataset
    dataset = []    # 字典构成的列表 [{'input':[ids], 'target': id},{}]
    # 遍历所有句子的id列表
    for sentence_id_list in sentences_id:
        # 遍历每一个id
        for i in range(len(sentence_id_list) - SEQ_LEN):
            # 每5个构成一个input，后面的是target
            input = sentence_id_list[i:i+SEQ_LEN]
            target = sentence_id_list[i+SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def preprocess():
    print("-------开始数据预处理...-------")

    # 1. 读取JSON文件，得到DataFrame；并做随机抽样
    df = pd.read_json(RAW_DATA_DIR / RAW_DATA_FILE, lines=True, orient='records').sample(frac=0.1)

    # 2. 提取所有对话句子，并做清洗
    sentences = []
    # 遍历所有组对话
    for dialog in df['dialog']:
        # 遍历本组对话中的每一句，并做处理
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(sentences[0])
    print(len(sentences))

    # 3. 对原始语料做划分
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    # 4. 分词并构建词表、保存到文件
    JiebaTokenizer.build_vocab(train_sentences, MODEL_DIR/VOCAB_FILE)

    # 5. 创建分词器
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR/VOCAB_FILE)

    # 6. 构建数据集
    train_dataset = build_dataset(train_sentences, tokenizer)
    test_dataset = build_dataset(test_sentences, tokenizer)

    # 7. 保存数据集到文件
    pd.DataFrame(train_dataset).to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE, orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print("-------数据预处理完成-------")


if __name__ == '__main__':
    preprocess()