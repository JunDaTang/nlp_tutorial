import jieba
from config import *

from nltk import TreebankWordTokenizer, TreebankWordDetokenizer     # 英文分词器

class BaseTokenizer():
    unk_token = UNK_TOKEN   # 类属性
    pad_token = PAD_TOKEN
    start_token = START_TOKEN
    end_token = END_TOKEN

    # 初始化
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)   # 词表大小
        self.word2id = { word : id for id, word in enumerate(vocab_list) }
        self.id2word = { id : word for id, word in enumerate(vocab_list) }
        # self.unk_token = UNK_TOKEN
        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]
        self.start_id = self.word2id[self.start_token]
        self.end_id = self.word2id[self.end_token]

    # 分词，类方法接口
    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    # 编码（将文本分词、id化），并指定序列长度
    def encode(self, text, mark=False):
        # 分词
        tokens = self.tokenize(text)

        # 如果是目标序列，就在前后加入标记
        if mark:
            tokens = [self.start_token] + tokens + [self.end_token]

        # id化
        ids = [self.word2id.get(token, self.unk_id) for token in tokens]
        return ids

    # 构建词表，并保存到文件
    @classmethod
    def build_vocab(cls, sentences, vocab_file_path):
        # 1. 针对训练集分词，构建词表
        vocab_set = set()  # 利用集合做token去重
        for sentence in sentences:
            vocab_set.update( cls.tokenize(sentence) )
        # 转换成列表（词表，id2word），并处理未登录词和填充词
        vocab_list = [cls.pad_token, cls.unk_token, cls.start_token, cls.end_token] + list(vocab_set)

        print("词表大小：", len(vocab_list))

        # 2. 保存词表到文件
        with open( vocab_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    # 从文件加载词表，并创建分词器对象实例
    @classmethod
    def from_vocab(cls, vocab_file_path):
        # 1. 获取词表
        with open( vocab_file_path, 'r', encoding='utf-8') as f:
            # 读取每一行
            vocab_list = [token.strip() for token in f.readlines()]
        # 2. 构建分词器对象
        tokenizer = cls(vocab_list)
        return tokenizer

# 定义子类
# 中文分词器
class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)

# 英文分词器
class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    # 解码：传入一个id列表，返回原始英文句子
    def decode(self, ids):
        # 将id转换为 token
        tokens = [ self.id2word[id] for id in ids ]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    en_tokenizer = EnglishTokenizer.from_vocab( MODEL_DIR / EN_VOCAB_FILE )
    cn_tokenizer = ChineseTokenizer.from_vocab( MODEL_DIR / CN_VOCAB_FILE )

    print("中文词表大小：", cn_tokenizer.vocab_size)
    print("英文词表大小：", en_tokenizer.vocab_size)
    print("特殊符号UNK：", cn_tokenizer.unk_token)
    print("特殊符号PAD ID：", en_tokenizer.pad_id)
    print("特殊符号START：", en_tokenizer.start_token)
    print("特殊符号END ID：", cn_tokenizer.end_id)

    print( cn_tokenizer.encode("自然语言处理") )
    print( en_tokenizer.encode("hello world!", mark=True) )
