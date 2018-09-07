import os

import re
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from DealtextUtil import DealtextUtil
from TaggedLineSentence import TaggedLineSentence


class TextclfModel(object):

    paths = ["D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\退款",
             "D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\未退款"]

    # def __init__(self,paths):
    #     self.paths=paths


    # LogisticRegression分类器
    def LRclassifier(self,train_arrays, train_labels, test_arrays, test_label):
        classifier = LogisticRegression(solver='sag')
        classifier.fit(train_arrays, train_labels)
        y = classifier.predict(test_arrays)
        print('accuracy', metrics.accuracy_score(test_label.astype(int), y))
        return classifier

    # 随机森林分类器
    def RFclassifier(self,train_arrays, train_labels, test_arrays, test_label):
        classifier = RandomForestClassifier(oob_score=True, random_state=10, max_depth=5, n_estimators=30)
        classifier.fit(train_arrays, train_labels.astype(int))
        print('oob', classifier.oob_score_)
        y = classifier.predict(test_arrays)
        print('accuracy', metrics.accuracy_score(test_label.astype(int), y))
        return classifier

    # KNN分类
    def Knnclassifier(self,train_arrays, train_labels, test_arrays, test_label):
        classifier = KNeighborsClassifier(n_neighbors=10, algorithm='brute', metric='cosine')
        classifier.fit(train_arrays, train_labels.astype(int))
        y = classifier.predict(test_arrays)
        print('accuracy', metrics.accuracy_score(test_label.astype(int), y))
        print('F1 score', metrics.f1_score(test_label.astype(int), y))
        return classifier

    # SVM分类器
    def SVMclassifier(self,train_arrays, train_labels, test_arrays, test_label):
        classifier = SVC(C=7, gamma=0.0001, kernel='rbf', class_weight='balanced')
        # classifier=LinearSVC(C=2,penalty='l2',class_weight='balanced',dual=False)
        classifier.fit(train_arrays, train_labels.astype(int))
        y = classifier.predict(test_arrays)
        # print('y123',y)
        a = np.c_[y, test_label.astype(int)]
        # print(a)
        print('accuracy', metrics.accuracy_score(test_label.astype(int), y))
        print('F1 score', metrics.f1_score(test_label.astype(int), y))
        return classifier

    # getauc
    def get_SVMauc(self,train_arrays, train_labels, test_arrays, test_label):
        # classifier = LinearSVC(C=2, penalty='l2', class_weight='balanced')
        classifier = SVC(C=7, gamma=0.0001, kernel='rbf', class_weight='balanced')
        classifier.fit(train_arrays, train_labels.astype(int))
        y = classifier.predict(test_arrays)
        # print('y123',y)
        a = np.c_[y, test_label.astype(int)]
        print('accuracy', metrics.accuracy_score(test_label.astype(int), y))
        return metrics.accuracy_score(test_label.astype(int), y)

    # getauc
    def get_LRauc(self,train_arrays, train_labels, test_arrays, test_label):
        classifier = RandomForestClassifier(oob_score=True, random_state=10, max_depth=5, n_estimators=30)
        classifier.fit(train_arrays, train_labels.astype(int))
        print('oob', classifier.oob_score_)
        y = classifier.predict(test_arrays)
        print('accuracy', metrics.accuracy_score(test_label.astype(int), y))
        return metrics.accuracy_score(test_label.astype(int), y)

    # 得到doc2vec模型
    def get_model(self,paths, i=80, j=6):
        tls = TaggedLineSentence()
        tagged_stc = tls.to_array(paths)
        model = Doc2Vec(min_count=1, vector_size=100, window=10, sample=1e-4, negative=j, dm=0, workers=3)
        model.build_vocab(tagged_stc)
        for epoch in range(i):
            model.train(tls.perm(),
                        total_examples=model.corpus_count,
                        epochs=model.iter)
        model.save('./model/training_model/fyfl_pc.d2v')

    # 调参用
    def get_model2(self,paths, i=80, j=6):
        tls = TaggedLineSentence()
        tagged_stc = tls.to_array(paths)
        model = Doc2Vec(min_count=1, vector_size=100, window=15, sample=1e-4, negative=j, dm=0, workers=3)
        model.build_vocab(tagged_stc)
        for epoch in range(i):
            model.train(tls.perm(),
                        total_examples=model.corpus_count,
                        epochs=model.iter)
        return model

    # 绘图显示模型nsample和auc的关系,i,j模型参数epochs和 nsample
    def display_auc_ns(self, i=80):
        auc = []
        auc2 = []
        xlabels = []
        dtu = DealtextUtil()
        c = dtu.get_filecount(self.paths)
        pc_count = c[self.paths[0]]
        nopc_count = c[self.paths[1]]
        for j in range(5, 21):
            xlabels.append(j)
            model = self.get_model2(self.paths, i, j)

            pc = np.zeros((pc_count, 100))
            pc_label = np.zeros(pc_count)
            nopc = np.zeros((nopc_count, 100))
            nopc_labels = np.zeros(nopc_count)

            for i in range(pc_count):
                pc_pos = 'pc_' + str(i)
                pc[i] = model.docvecs[pc_pos]
                pc_label[i] = 1

            for i in range(nopc_count):
                nopc_pos = 'nopc_' + str(i)
                nopc[i] = model.docvecs[nopc_pos]
                nopc_labels[i] = 0

            X = np.r_[pc, nopc]
            Y = np.r_[pc_label, nopc_labels]

            # 归一化
            X = preprocessing.scale(X)
            train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, Y, test_size=0.25,
                                                                                    random_state=5)
            accuracy = self.get_SVMauc(train_arrays, train_labels, test_arrays, test_labels)
            # accuracy2=get_LRauc(train_arrays,train_labels,test_arrays,test_labels)
            auc.append(accuracy)
            # auc2.append(accuracy2)
        plt.figure()
        plt.title('auc-epochs')
        plt.xlabel(xlabels)
        plt.plot(xlabels, auc, 'b--')
        # plt.plot(xlabels,auc2,'r--')
        plt.show()

    # 绘图显示模型epoch和auc的关系
    def display_auc_ep(self, j=6):
        auc = []
        auc2 = []
        xlabels = []
        dtu = DealtextUtil()
        c = dtu.get_filecount(self.paths)
        pc_count = c[self.paths[0]]
        nopc_count = c[self.paths[1]]
        for i in range(0, 201, 10):
            xlabels.append(i)
            model = self.get_model2(self.paths, i, j)
            pc = np.zeros((pc_count, 100))
            pc_label = np.zeros(pc_count)
            nopc = np.zeros((nopc_count, 100))
            nopc_labels = np.zeros(nopc_count)

            for i in range(pc_count):
                pc_pos = 'pc_' + str(i)
                pc[i] = model.docvecs[pc_pos]
                pc_label[i] = 1

            for i in range(nopc_count):
                nopc_pos = 'nopc_' + str(i)
                nopc[i] = model.docvecs[nopc_pos]
                nopc_labels[i] = 0

            X = np.r_[pc, nopc]
            Y = np.r_[pc_label, nopc_labels]

            # 归一化
            X = preprocessing.scale(X)
            train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, Y, test_size=0.25,
                                                                                    random_state=5)
            accuracy = self.get_SVMauc(train_arrays, train_labels, test_arrays, test_labels)
            # accuracy2=get_LRauc(train_arrays,train_labels,test_arrays,test_labels)
            auc.append(accuracy)
            # auc2.append(accuracy2)
        plt.figure()
        plt.title('auc-epochs')
        plt.xlabel(xlabels)
        plt.plot(xlabels, auc, 'b--')
        # plt.plot(xlabels,auc2,'r--')
        plt.show()

    # 获取模型,todo：后面可以改一下增加复用性
    def get_clfmodel(self):
        X, Y = self.get_data()
        # 归一化
        X = preprocessing.scale(X)
        train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, Y, test_size=0.25, random_state=37)
        classifier1 = self.SVMclassifier(train_arrays, train_labels, test_arrays, test_labels)
        # classifier1 = Knnclassifier(train_arrays, train_labels, test_arrays, test_labels)
        joblib.dump(classifier1, './model/training_model/clfmodel.m')

    # 获取数据
    def get_data(self):
        model = Doc2Vec.load('./model/fyfl_pc.d2v')
        dtu = DealtextUtil()
        c = dtu.get_filecount(self.paths)
        pc_count = c[self.paths[0]]
        nopc_count = c[self.paths[1]]
        pc = np.zeros((pc_count, 100))
        pc_label = np.zeros(pc_count)
        nopc = np.zeros((nopc_count, 100))
        nopc_labels = np.zeros(nopc_count)

        for i in range(pc_count):
            pc_pos = 'pc_' + str(i)
            pc[i] = model.docvecs[pc_pos]
            pc_label[i] = 1

        for i in range(nopc_count):
            nopc_pos = 'nopc_' + str(i)
            nopc[i] = model.docvecs[nopc_pos]
            nopc_labels[i] = 0

        X = np.r_[pc, nopc]
        Y = np.r_[pc_label, nopc_labels]
        return X, Y

    # 标准化外部向量
    def get_prodata(self, vec):
        X, y = self.get_data()
        # X=np.array(X).reshape(-1,100)
        scaler = preprocessing.StandardScaler().fit(X)
        vec = np.array(vec).reshape(-1, 100)
        vec = scaler.transform(vec)
        return vec

    # 归一化和非归一化的auc比较
    def draw_auc_pro(self):
        X, Y = self.get_data()
        print(X.shape)
        # 归一化
        auclist = []
        auc2list = []
        xlabels = []
        X1 = preprocessing.scale(X)
        for i in range(50):
            train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, Y, test_size=0.25,
                                                                                    random_state=i)
            auc = self.get_SVMauc(train_arrays, train_labels, test_arrays, test_labels)
            auclist.append(auc)
            train_arrays, test_arrays, train_labels, test_labels = train_test_split(X1, Y, test_size=0.25,
                                                                                    random_state=i)
            auc2 = self.get_SVMauc(train_arrays, train_labels, test_arrays, test_labels)
            auc2list.append(auc2)
            xlabels.append(i)
        plt.figure()
        plt.plot()

        plt.plot(xlabels, auclist, 'b--')
        plt.plot(xlabels, auc2list, 'r--')
        plt.show()

    # 用正则匹配去分类文件，效果并不是很好，暂时不用了
    def judge_zs(self,text):
        pattern = re.compile('退赔\w*银行|退还\w*银行|归还\w*银行|退缴\w*银行|退赔\w*欠款'
                             '|退还\w*欠款|归还\w*欠款|退缴\w*欠款|赔偿\w*欠款|赔偿\w*银行|'
                             '退赔\w*本金|退还\w*本金|归还\w*本金|退缴\w*本金|退赔\w*本金')
        result = re.search(pattern, text)
        if result != None:
            return 1
        else:
            return 0

    def judge_zs2(self,text):
        pattern = re.compile('未退赔|未退还|未归还|未退缴')
        result = re.search(pattern, text)
        if result != None:
            return 1
        else:
            return 0

    # 计算准确率
    def eval_auc(self, paths):
        zs_count = len(os.listdir(paths[0]))
        fzs_count = len(os.listdir(paths[1]))
        all_count = zs_count + fzs_count
        i = 0
        j = 0
        files = os.listdir(paths[0])
        dtu = DealtextUtil()
        for file in files:
            file = os.path.join(paths[0], file)
            text = dtu.readdoc2(file)
            result = self.judge_zs(text)
            if result == 1:
                i += 1
        files = os.listdir(paths[1])
        for file in files:
            file = os.path.join(paths[1], file)
            text = dtu.readdoc2(file)
            result = self.judge_zs(text)
            if result == 0:
                j += 1
        print(i)
        print(j)
        auc = float((i + j) / all_count)
        print(auc)

    #GridSerachcv
    def Grid(self):
        X, y = self.get_data()
        auc = []
        # 归一化
        X = preprocessing.scale(X)
        train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=37)

        C = [1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]
        gamma = [0.001, 0.0001]
        param_test = dict(C=C, gamma=gamma)
        clf = SVC(kernel='rbf', class_weight='balanced')
        gridSVC = GridSearchCV(clf, param_grid=param_test, cv=5, scoring='accuracy')
        gridSVC.fit(train_arrays, train_labels)
        print('best score is:', str(gridSVC.best_score_))
        print('best params are', str(gridSVC.best_params_))

    #交叉验证
    def ModelCV(self,model):
        X,y=self.get_data()
        X=preprocessing.scale(X)
        train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=24)
        kfold = KFold(n_splits=10, shuffle=True, random_state=37)
        scores2 = cross_val_score(model, X, y, cv=kfold)
        scores2=np.mean(scores2)
        return scores2

    #加载分类模型
    def loadclf(self,modelpath):
        model=joblib.load(modelpath)
        return model


