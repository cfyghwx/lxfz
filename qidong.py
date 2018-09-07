from Doc222Vector import Doc222Vector
from LXResult import LXResult
from TextclfModel import TextclfModel
from RegressionModel import RegressionModel

if __name__=='__main__':
    # res=LXResult()
    # res.Getresult("D:\\南京大学\\天津方面事务\\自动文本分类\\数据记录\\newdata\\测试_new\\未赔偿\\6.docx")
    # rem=RegressionModel()
    # rem.save_model_je()
    # rem.save_model_xq()
    d=Doc222Vector()
    d.get_tesresult()
