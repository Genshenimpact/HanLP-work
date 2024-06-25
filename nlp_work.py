from pyhanlp import *
import pandas as pd

df = pd.read_csv('data/usual_eval_labeled.csv', encoding='GBK')
# 定义分词函数
def segment_text(text):
    seg_next = HanLP.segment(text)
    segmented_text = ' '.join([term.word for term in seg_next])
    return segmented_text

# 对文本进行分词处理
df['segmented_text'] = df['text'].apply(segment_text)

# 保存处理后的数据到新的CSV文件
df.to_csv('output_file.utf8', index=False, encoding='utf8')

FirstOrderMEKf = JClass('com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModel')
HMMSegment = JClass('com.hankcs.hanlp.model.hmm.HMMSegmenter')
segmenter = HMMSegment(FirstOrderMEKf())
segmenter.train('E:/nlp/output_file.utf8')


def bmes_tagging(seg_list):
    tagged_list = []
    for word in seg_list:
        if len(word) == 1:
            tagged_list.append((word, 'S'))
        else:
            tagged_list.append((word[0], 'B'))
            for char in word[1:-1]:
                tagged_list.append((char, 'M'))
            tagged_list.append((word[-1], 'E'))
    return tagged_list

df_test = pd.read_csv('data/usual_test_labeled.csv')['文本']

all_Num=0
right_Num = 0
for text in df_test:
    seg_list = segmenter.segment(text)
    my_tag = bmes_tagging(seg_list)
    HanLP.Config.ShowTermNature = False
    org = HanLP.segment(text)
    fucl = segment_text(text)
    word_list = fucl.split(" ")
    han_word_tag = bmes_tagging(word_list)
    print(my_tag)
    print(han_word_tag)
    for i in han_word_tag:
        for j in my_tag:
            if i == j:
                right_Num = right_Num + 1
                break
        all_Num = all_Num + 1
acc = right_Num/all_Num
print(acc)


def try_load_model(self, trained):

    if trained:
        with open(self.model_file, 'rb') as f:
            self.A_dic = pickle.load(f)
            self.B_dic = pickle.load(f)
            self.Pi_dic = pickle.load(f)
            self.load_para = True
    else:
        # 状态转移概率 P(ok | ok-1)
        self.A_dic = {}
        # 发射概率 P(λk | ok)
        self.B_dic = {}
        # 状态的初始概率 P(λ1 = ok)
        self.Pi_dic = {}
        self.load_para = False
