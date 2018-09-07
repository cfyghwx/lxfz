from Content import Content
from RegressionModel import RegressionModel


class LXResult(object):

    def Getresult(self,path):
        c=Content()
        vec=c.get_inputv(path)
        inputv=vec[:,2:]
        print(inputv)
        rmodel=RegressionModel()
        a, b = rmodel.get_result_outer(inputv)

        a = a[0]
        b = b[0]
        print('预测罚金', a[0], '预测刑期', a[1])
        print('建议罚金范围', b[0], '建议量刑范围', b[1])



