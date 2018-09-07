import os

from gensim.models import Doc2Vec
from DealtextUtil import DealtextUtil
import matplotlib.pyplot as plt
import numpy as np
from TextclfModel import TextclfModel


class Doc222Vector(object):

    #将文本转化为向量
    def get_vec(self,path):
        dtu=DealtextUtil()
        doc_word, textc = dtu.readdoc3(path)
        model = Doc2Vec.load('./model/fyfl_pc.d2v')
        vec = model.infer_vector(doc_word, alpha=0.025, steps=140)  # 0.025，140
        return vec, textc


    #用作调参用,i为alpha，j为steps
    def get_vec_tc(self,path,i=0.025,j=140):
        dtu=DealtextUtil()
        doc_word, textc = dtu.readdoc3(path)
        model = Doc2Vec.load('./model/fyfl_pc.d2v')
        vec = model.infer_vector(doc_word, alpha=i, steps=j)  # 0.025，140
        return vec


    def get_pengchang(self,path,i=0.025,j=140):
        vec = self.get_vec_tc(path,i,j)
        clfmodel=TextclfModel()
        clf=clfmodel.loadclf('./model/clfmodel.m')
        vec = clfmodel.get_prodata(vec)
        vec = np.array(vec).reshape(-1, 100)
        y = clf.predict(vec)
        return y

    def tcfun_alpha(self,j=140):

        paths = ["D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\退款",
                 "D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\未退款"]

        all_count = len(os.listdir(paths[0])) + len(os.listdir(paths[1]))
        pc_count = 0
        nopc_count = 0
        steps = np.linspace(0, 0.1, 20)
        # steps=[200]
        auclist = []
        for i in steps:
            # 统计1
            files = os.listdir(paths[0])
            for file in files:
                file = os.path.join(paths[0], file)
                label = self.get_pengchang(file,i,j)
                if label == 1:
                    pc_count += 1
                    # print(pc_count)
            files = os.listdir(paths[1])
            for file in files:
                file = os.path.join(paths[1], file)
                label= self.get_pengchang(file,i,j)
                if label == 0:
                    nopc_count += 1
                    # print(nopc_count)
            auc = float((pc_count + nopc_count) / all_count)
            auclist.append(auc)
            print(auc)
            pc_count = 0
            nopc_count = 0
        plt.figure()
        plt.title('alpha 0-0.1')
        plt.plot(steps, auclist)
        plt.show()
        print(auclist)

    def tcfun_step(self,i=0.025):

        # paths = ["D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\退款",
        #          "D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\未退款"]

        paths = ["D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\测试\\赔偿",
             "D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\测试\\未赔偿"]
        all_count = len(os.listdir(paths[0])) + len(os.listdir(paths[1]))
        pc_count = 0
        nopc_count = 0
        steps = range(0,201,10)
        auclist = []
        for j in steps:
            # 统计1
            files = os.listdir(paths[0])
            for file in files:
                file = os.path.join(paths[0], file)
                label= self.get_pengchang(file,i,j)
                if label == 1:
                    pc_count += 1
                    # print(pc_count)
            files = os.listdir(paths[1])
            for file in files:
                file = os.path.join(paths[1], file)
                label = self.get_pengchang(file,i,j)
                if label == 0:
                    nopc_count += 1
                    # print(nopc_count)
            auc = float((pc_count + nopc_count) / all_count)
            auclist.append(auc)
            print(auc)
            pc_count = 0
            nopc_count = 0
        plt.figure()
        plt.title('alpha 0-0.1')
        plt.plot(steps, auclist)
        plt.show()
        print(auclist)

    def get_tesresult(self):

        paths = ["D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\测试_new\\赔偿",
                 "D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\测试_new\\未赔偿"]

        files = os.listdir(paths[0])
        count1 = len(files)
        i = 0
        for file in files:
            file = os.path.join(paths[0], file)
            label=self.get_pengchang(file)
            if label == 1:
                i += 1
        a = float(i / count1)
        print('对1的准确率：', float(i / count1))
        files = os.listdir(paths[1])
        count2 = len(files)
        j = 0
        for file in files:
            file = os.path.join(paths[1], file)
            label = self.get_pengchang(file)
            if label == 0:
                j += 1
        b = float(j / count2)
        print('对0的准确率：', float(j / count2))
        print('准确率：', float((i + j) / (count1 + count2)))
        print('准确率2：', float((a + b) / 2))