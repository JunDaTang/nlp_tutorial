# 数据预处理

from config import *

from datasets import load_dataset, ClassLabel  # 加载数据集
from transformers import AutoTokenizer  # 分词器


def preprocess():
    print("-------开始数据预处理...-------")

    # 1. 读取csv文件，得到字典，提取Dataset
    dataset = load_dataset('csv', data_files=str(RAW_DATA_DIR/RAW_DATA_FILE))['train']
    # print(dataset)

    # 2. 去掉cat列，数据过滤
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter( lambda x: x['review'] is not None )
    # print(dataset)

    # 3. 对原始语料做划分，按label分层抽样
    dataset = dataset.cast_column('label', ClassLabel(names=['n', 'p']))
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    # print(dataset_dict)

    # 4. 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_DIR/BERT_MODEL)

    # 5. 构建数据集
    def batch_encode(example):
        inputs = tokenizer(
            example['review'],
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        # 添加标签字段labels
        inputs['labels'] = example['label']
        return inputs

    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['label', 'review'])
    print(dataset_dict)

    # 6. 保存数据集到文件
    dataset_dict.save_to_disk( PROCESSED_DATA_DIR )

    print("-------数据预处理完成-------")


if __name__ == '__main__':
    preprocess()