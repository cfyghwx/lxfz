import os
import random
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from  DealtextUtil import DealtextUtil

#处理文本，将文本分词并转化并标注
class TaggedLineSentence(object):
    # def __init__(self,sources):
    #     self.sources=sources

    def to_array(self,paths):
        self.sentences = []
        dtu = DealtextUtil()
        for path in paths:
            # print(path)
            files=os.listdir(path)
            for file in files:
                content,pos=dtu.get_doclist(path,file)
                # print(len(content))
                self.sentences.append(TaggedDocument(content,[pos]))
        return self.sentences

    def perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled




