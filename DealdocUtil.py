import os
from win32com import client

#主要用于处理doc文件，包含将doc转换为docx，改变文件名称
class DealdocUtil(object):

    # 将下载的doc和docx统一转化为docx，为之后解析文本内容做准备
    def deal_doc(path):
        files = os.listdir(path)
        # print(files)
        word = client.Dispatch("Word.Application")
        i = 0
        for file in files:
            file = path + '\\' + file
            doc = word.Documents.Open(file);
            str1 = path + '\\' + str(i)
            doc.SaveAs(str1, 12)
            os.remove(file)
            i += 1
        doc.Close()
        word.Quit()

    #重命名已赔偿文件
    def pc_renamedoc(path):
        files = os.listdir(path)

        for file in files:
            filename = os.path.splitext(file)[0]
            filetype = os.path.splitext(file)[1]
            if filename:
                newfile = 'pc_' + filename  # pc=赔偿
                os.rename(os.path.join(path, filename + filetype), os.path.join(path, newfile + filetype))

    # 重命名未赔偿的文件
    def nopc_renamedoc(path):
        files = os.listdir(path)
        j = 0
        for file in files:
            filename = os.path.splitext(file)[0]
            filetype = os.path.splitext(file)[1]
            if filename:
                newfile = 'nopc_' + filename  # pc=赔偿
                os.rename(os.path.join(path, filename + filetype), os.path.join(path, newfile + filetype))

    #重命名自首文件
    def zs_renamedoc(path):
        files = os.listdir(path)

        for file in files:
            filename = os.path.splitext(file)[0]
            filetype = os.path.splitext(file)[1]
            if filename:
                newfile = 'zs_' + filename  # zs=自首
                os.rename(os.path.join(path, filename + filetype), os.path.join(path, newfile + filetype))

    # 重命名非自首的文件
    def nozs_renamedoc(path):
        files = os.listdir(path)
        j = 0
        for file in files:
            filename = os.path.splitext(file)[0]
            filetype = os.path.splitext(file)[1]
            if filename:
                newfile = 'nozs_' + filename  # 非自首
                os.rename(os.path.join(path, filename + filetype), os.path.join(path, newfile + filetype))
