import os
import sys
from torchtext import data, datasets
import sentencepiece as spm
import pandas as pd
import MeCab

PAD = -1
BOS = 2
EOS = 3

def text_tokenizer(): 
    
    m = MeCab.Tagger()
    delete_tag = ['BOS/EOS', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']

    
    
    def remove_josa(sentence):
        sentence_split = sentence.split() # 원본 문장 띄어쓰기로 분리

        dict_list = []

        for token in sentence_split: # 띄어쓰기로 분리된 각 토큰 {'단어':'형태소 태그'} 와 같이 딕셔너리 생성
            m.parse('')
            node = m.parseToNode(token)
            word_list = []
            pos_list = []
        
            while node:
                morphs = node.feature.split(',')
                word_list.append(node.surface)
                pos_list.append(morphs[0])
                node = node.next
            dict_list.append(dict(zip(word_list, pos_list)))        

        for dic in dict_list: # delete_tag에 해당하는 단어 쌍 지우기 (조사에 해당하는 단어 지우기)
            for key in list(dic.keys()):
                if dic[key] in delete_tag:
                    del dic[key]

        combine_word = [''.join(list(dic.keys())) for dic in dict_list] # 형태소로 분리된 각 단어 합치기
        result = ' '.join(combine_word) # 띄어쓰기로 분리된 각 토큰 합치기

        return result # 온전한 문장을 반환
    
    
    
    
    df = pd.read_csv("test.csv")
    KOR_data = df['Korean']
    
    f = open("kor_no_josa.txt", "w", encoding = 'utf-8')
    for row in KOR_data[:100000]:
        f.write(remove_josa(row)) 
        f.write('\n')
    f.close()
    
    spm.SentencePieceTrainer.Train('--input=kor_no_josa.txt \
                               --model_prefix=korean_tok \
                               --vocab_size=100000 \
                               --hard_vocab_limit=false')
    
    sp = spm.SentencePieceProcessor()
    sp.Load('korean_tok.model')
    
    return lambda x : sp.EncodeAsPieces(x)
 
def English_tokenizer():
    
    df = pd.read_csv("test.csv")
    ENG_data = df['English']
    
    f = open("eng.txt", "w", encoding = 'utf-8')
    for row in ENG_data[:100000]:
        f.write(row)
        f.write('\n')
    f.close()
    
    spm.SentencePieceTrainer.Train('--input=eng.txt \
                               --model_prefix=english_tok \
                               --vocab_size=100000\
                               --hard_vocab_limit=false')
    
    sp = spm.SentencePieceProcessor()
    sp.Load('english_tok.model')
    
    return lambda x : sp.EncodeAsPieces(x)

class DataLoader():
  
    def __init__(self, train_fn = None, 
                    valid_fn = None, 
                    exts = None,
                    batch_size = 64, 
                    device = 'cpu', 
                    max_vocab = 99999999,    
                    max_length = 255, 
                    fix_length = None, 
                    use_bos = True, 
                    use_eos = True, 
                    shuffle = True
                    ):

        super(DataLoader, self).__init__()
        
        # text(source) 데이터의 틀
        self.src = data.Field(sequential = True,
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = None, 
                                eos_token = None,
                                tokenize = Korean_tokenizer()
                                )
        super(DataLoader, self).__init__()
        
        # code(target) 데이터의 틀
        self.tgt = data.Field(sequential = True, 
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = '<BOS>' if use_bos else None, 
                                eos_token = '<EOS>' if use_eos else None,
                                tokenize = English_tokenizer()
                                )
        
        
        # train, test(valid) 데이터 틀(앞서 생성한 src, tgt 각각)
        train = TranslationDataset(path = train_fn, exts = exts,
                                        fields = [('src', self.src), ('tgt', self.tgt)], 
                                        max_length = max_length
                                        )
        valid = TranslationDataset(path = valid_fn, exts = exts,
                                        fields = [('src', self.src), ('tgt', self.tgt)], 
                                        max_length = max_length
                                        )
        
        # 각각의 epoch에 대해 새로 섞인 batch를 생성하면서 반복
        # train, test(valid)를 각각 나눴으니 아래 작업은 진행해도 무방할 듯
        self.train_iter = data.BucketIterator(train, 
                                                batch_size = batch_size, 
                                                shuffle = shuffle, 
                                                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), 
                                                sort_within_batch = True
                                                )

        self.valid_iter = data.BucketIterator(valid, 
                                                batch_size = batch_size, 
                                                shuffle = False, 
                                                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), 
                                                sort_within_batch = True
                                                )
        
        self.src.build_vocab(train, max_size = max_vocab)
        self.tgt.build_vocab(train, max_size = max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab
        

class TranslationDataset(data.Dataset):

    def sort_key(ex):  # 음수와 양수 모두 가능
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        print(exts)
        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        
        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()

                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)
 
        
if __name__ == '__main__':

     """
     argv1,2 : src.csv와 tgt.csv파일이 있는 공통 경로
     (argv3, argv4) : 확장자를 포함한 각 파일 이름
     """
    loader = DataLoader('C:/Users/USER/Chambit/','C:/Users/USER/Chambit/' , ('text_sample.csv','code_sample.csv'),
                         shuffle = False, 
                         batch_size = 8
                         )
    
    
    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))
    
    for batch_index, batch in enumerate(loader.train_iter):
        print(batch_index)
        print(batch.src)
        print(batch.tgt)
        
        if batch_index > 1:
            break
