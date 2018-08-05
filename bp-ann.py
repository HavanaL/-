from tkinter import *
from tkinter import Menu
from tkinter import filedialog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from tkinter import ttk
from tkinter import Spinbox
from sklearn.externals import joblib
import matplotlib
from sklearn import preprocessing


class demo(object):
    def __init__(self):
        self.init_window = Tk()
        self.init_window.title('茶鲜叶产地判别')
        menubar = Menu(self.init_window)
        self.init_window.config(menu=menubar,borderwidth=3,bg='lightblue')
        self.Openmenu = Menu(menubar, tearoff=0, fg='white', bg='lightblue')
        self.secondmenu = Menu(self.Openmenu, tearoff=0, fg='white', bg='lightblue')
        menubar.add_cascade(label='数据',menu=self.Openmenu)
        self.Openmenu.add_separator()
        self.Openmenu.add_cascade(label='训练集', menu=self.secondmenu)
        self.Openmenu.add_cascade(label='测试集', menu=self.secondmenu)
        self.Openmenu.add_command(label='保存文件',command=self.save)
        self.secondmenu.add_command(label='显示数据', command=self.display)
        self.secondmenu.add_command(label='绘制图表', command=self.graph)
        self.init_Text = Text(self.init_window, width=180, height=50)
        self.init_Text.grid(row=0, column=20, rowspan=15, columnspan=10)

        self.dealmenu = Menu(self.Openmenu, tearoff=0, fg='white', bg='lightblue')
        menubar.add_cascade(label='预处理', menu=self.dealmenu)
        self.dealmenu.add_separator()
        self.dealmenu.add_command(label='PCA', command=self.pcadata)
        self.dealmenu.add_command(label='标准化',command=self.scale)
        self.dealmenu.add_command(label='归一化',command=self.normalize)

        self.modelingmenu = Menu(self.Openmenu, tearoff=0, fg='white', bg='lightblue')
        menubar.add_cascade(label='模型', menu=self.modelingmenu)
        self.modelingmenu.add_command(label='模型训练', command=self.createmodel)
        self.modelingmenu.add_command(label='保存模型',command=self.savemodel)

        self.judgemenu=Menu(self.Openmenu,tearoff=0,fg='white',bg='lightblue')
        self.secondjudgemenu = Menu(self.Openmenu, tearoff=0, fg='white', bg='lightblue')
        menubar.add_cascade(label='判别产地',menu=self.judgemenu)
        self.judgemenu.add_command(label='选择模型',command=self.selectmodel)
        self.judgemenu.add_cascade(label='判别',menu=self.secondjudgemenu)
        self.secondjudgemenu.add_command(label='判别一个',command=self.inputlayer)
        self.secondjudgemenu.add_command(label='判别所有',command=self.inputlayer1)

        self.deleterbutton=Button(self.init_window,text='清空内容',command=self.deleter,bg='white',fg='black',width=10)
        self.deleterbutton.grid(row=0,column=30)
        self.init_window.mainloop()
    def deleter(self):
         self.init_Text.delete(1.0,END)

    def display(self):
        self.x = filedialog.askopenfile()
        self.c = np.array(pd.read_excel(self.x.name))
        np.set_printoptions(threshold=1000000000)
        self.init_Text.delete(1.0, END)
        self.init_Text.insert(1.0, self.c)

    def graph(self):
        file = filedialog.askopenfile()
        data = np.array(pd.read_excel(file.name))
        plt.figure()
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.xlabel('波长/cm-1')
        plt.ylabel('吸收值')
        x_axis = data.T[0]
        for i in range(1, len(data.T)):
            y_axis = data.T[i]
            plt.plot(x_axis, y_axis)
        plt.show()

    def save(self):
        self.savefile=filedialog.asksaveasfilename(filetypes=[('npy文件','*.npy')],initialfile='data')
        np.save(self.savefile,self.pcadata)

    def pcadata(self):
        self.setting = Toplevel()
        self.setting.title('设置')
        self.setwindowentry = Entry(self.setting, textvariable=StringVar)
        self.setwindowentry.grid(row=0, column=1)
        self.label = Label(self.setting, text='请输入降低的维度')
        self.label.grid(row=0, column=0)
        self.surebutton = Button(self.setting, text='确定', command=self.ensure)
        self.surebutton.grid(row=2, column=2)

    def ensure(self):
        self.dealfile = filedialog.askopenfile()
        self.dealdata =np.array(pd.read_excel(self.dealfile.name))
        pca = PCA(n_components=int(self.setwindowentry.get()))
        self.pcadata = pca.fit_transform((self.dealdata.T)[1:])
        self.init_Text.delete(1.0, END)
        self.init_Text.insert(1.0, self.pcadata)
    def scale(self):
        self.scalefile=filedialog.askopenfilename()
        self.scaledata=np.array(pd.read_excel(self.scalefile))
        self.x_scale=preprocessing.scale((self.scaledata.T)[1:])
        np.set_printoptions(threshold=1000000)
        self.init_Text.delete(1.0, END)
        self.init_Text.insert(1.0, self.x_scale)
    def normalize(self):
        self.normalizefile = filedialog.askopenfilename()
        self.normalizedata = np.array(pd.read_excel(self.normalizefile))
        self.x_normalize=preprocessing.normalize((self.normalizedata.T)[1:])
        np.set_printoptions(threshold=1000000)
        self.init_Text.delete(1.0, END)
        self.init_Text.insert(1.0, self.x_normalize)

    def createmodel(self):
        self.setparameter = Toplevel()
        self.setparameter.title('参数设置')
        self.comboslist = ttk.Combobox(self.setparameter, textvariable=StringVar)
        self.comboslist['values'] = ('logistic', 'relu', 'tanh', 'identity')
        self.comboslistlabel = Label(self.setparameter, text='激活函数')
        self.comboslistlabel.grid(row=4, column=0)
        self.comboslist.current(0)
        self.comboslist.grid(row=4, column=1)
        self.inputlayerlabel = Label(self.setparameter, text='输入层节点个数')
        self.inputlayer = Spinbox(self.setparameter, from_=1, to=100)
        self.inputlayer.grid(row=0, column=1)
        self.inputlayerlabel.grid(row=0, column=0)
        self.hiddenlayer = Spinbox(self.setparameter, from_=1, to=100)
        self.hiddenlayer.grid(row=1, column=1)
        self.hiddenlayerlabel = Label(self.setparameter, text='隐含层节点个数')
        self.hiddenlayerlabel.grid(row=1, column=0)
        self.outputlayer = Spinbox(self.setparameter, from_=1, to=100)
        self.outputlayer.grid(row=3, column=1)
        self.outputlayerlabel = Label(self.setparameter, text='输出层节点个数')
        self.outputlayerlabel.grid(row=3, column=0)
        self.learn_rate = Spinbox(self.setparameter, from_=0, to=1,increment=0.001)
        self.learn_rate.grid(row=0,column=3)
        self.learn_rate_label=Label(self.setparameter,text='学习率')
        self.learn_rate_label.grid(row=0,column=2)
        self.max_iter=Spinbox(self.setparameter,from_=1,to=10000)
        self.max_iter.grid(row=1,column=3)
        self.max_iter_label=Label(self.setparameter,text='最大更迭次数')
        self.max_iter_label.grid(row=1,column=2)
        self.surebutton=Button(self.setparameter,text='确定',command=self.importdata,bg='white',fg='lightblue',width=10)
        self.surebutton.grid(row=4,column=3)
    def importdata(self):
        self.testdata=Toplevel()
        self.trainlabel=Label(self.testdata,text='导入训练集数据')
        self.trainlabel.grid(row=0,column=0)
        self.labeldata=Label(self.testdata,text='导入标签值')
        self.labeldata.grid(row=1,column=0)
        self.traintext=Text(self.testdata,height=1,width=20)
        self.traintext.grid(row=0,column=1)
        self.labeltext=Text(self.testdata,height=1,width=20)
        self.labeltext.grid(row=1,column=1)
        self.ensurebutton1=Button(self.testdata,text='导入',command=self.daoru1)
        self.ensurebutton1.grid(row=0,column=2)
        self.ensurebutton2=Button(self.testdata,text='导入',command=self.daoru2)
        self.ensurebutton2.grid(row=1,column=2)
        self.ensurebutton3=Button(self.testdata,text='确定',command=self.sure,bg='white',fg='lightblue',width=10)
        self.ensurebutton3.grid(row=2,column=1)
    def daoru1(self):
        self.jiaozheng=filedialog.askopenfilename()
        self.traintext.insert(1.0,self.jiaozheng)
        self.str=np.load(self.jiaozheng)
        self.init_Text.delete(1.0,END)
        self.init_Text.insert(1.0,self.str)
    def daoru2(self):
        self.biaoqian = filedialog.askopenfilename()
        self.labeltext.insert(1.0, self.biaoqian)
        self.biaoqianvalue=np.array(pd.read_excel(self.biaoqian,header=None))
        self.init_Text.delete(1.0,END)
        self.init_Text.insert(1.0,self.biaoqianvalue)
    def sure(self):
        self.clf = MLPClassifier(activation=self.comboslist.get(), hidden_layer_sizes=(int(self.hiddenlayer.get()),),
                            learning_rate_init=float(self.learn_rate.get()), max_iter=int(self.max_iter.get()))
        self.clf.fit(self.str,self.biaoqianvalue)
        y=self.clf.predict(self.str)
        plt.figure()
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        x_axis = self.biaoqianvalue.T[0]
        y_axis = y
        plt.plot(x_axis, y_axis)
        plt.show()
    def savemodel(self):
        self.savefile = filedialog.asksaveasfilename(filetypes=[('pkl文件', '*.pkl')], initialfile='model.pkl')
        joblib.dump(self.clf,self.savefile)
    def inputlayer(self):
        self.inputlayernum=Toplevel()
        self.inplayerentry = Entry(self.inputlayernum, textvariable=StringVar)
        self.inplayerentry.grid(row=0, column=1)
        self.label = Label(self.inputlayernum, text='请输入所选模型的输入节点个数')
        self.label.grid(row=0, column=0)
        self.surebutton = Button(self.inputlayernum, text='确定', command=self.judgedata)
        self.surebutton.grid(row=2, column=2)
    def inputlayer1(self):
        self.inputlayernum=Toplevel()
        self.inplayerentry = Entry(self.inputlayernum, textvariable=StringVar)
        self.inplayerentry.grid(row=0, column=1)
        self.label = Label(self.inputlayernum, text='请输入所选模型的输入节点')
        self.label.grid(row=0, column=0)
        self.surebutton = Button(self.inputlayernum, text='确定', command=self.judgedata1)
        self.surebutton.grid(row=2, column=2)
    def judgedata(self):
        self.judgewindow = Toplevel()
        self.trainlabel = Label(self.judgewindow, text='导入测试集数据')
        self.trainlabel.grid(row=0, column=0)
        self.labeltext = Text(self.judgewindow, height=1, width=20)
        self.labeltext.grid(row=0, column=1)
        self.ensurebutton1 = Button(self.judgewindow, text='导入', command=self.daoru3)
        self.ensurebutton1.grid(row=0, column=2)
        self.trainlabel = Label(self.judgewindow, text='请输入样本编号')
        self.trainlabel.grid(row=1, column=0)
        self.labelentry = Entry(self.judgewindow,textvariable=StringVar)
        self.labelentry.grid(row=1, column=1)
        self.ensurebutton2 = Button(self.judgewindow, text='确定',command=self.judge,bg='white',fg='lightblue',width=10)
        self.ensurebutton2.grid(row=3, column=1)

    def judgedata1(self):
        self.judgewindow = Toplevel()
        self.trainlabel = Label(self.judgewindow, text='导入测试集数据')
        self.trainlabel.grid(row=0, column=0)
        self.labeltext = Text(self.judgewindow, height=1, width=20)
        self.labeltext.grid(row=0, column=1)
        self.ensurebutton1 = Button(self.judgewindow, text='导入', command=self.daoru3)
        self.ensurebutton1.grid(row=0, column=2)
        self.ensurebutton2 = Button(self.judgewindow, text='确定', command=self.judge1, bg='white', fg='lightblue',
                                    width=10)
        self.ensurebutton2.grid(row=3, column=1)
    def daoru3(self):
        self.ceshi=filedialog.askopenfilename()
        self.labeltext.insert(1.0,self.ceshi)
        self.str1=np.array(pd.read_excel(self.ceshi))
        self.yanzhen=(self.str1.T)[1:]
    def selectmodel(self):
        self.selectmodel = filedialog.askopenfilename()
        self.clfmodel = joblib.load(self.selectmodel)


    def judge(self):
        pca = PCA(n_components=int(self.inplayerentry.get()))
        self.testdata1=pca.fit_transform(self.yanzhen)
        self.init_Text.delete(1.0,END)
        self.init_Text.insert(1.0,self.clfmodel.predict(np.array(self.testdata1))[int(self.labelentry.get())])
        self.init_Text.insert(1.0, '产地为')
    def judge1(self):
        pca = PCA(n_components=int(self.inplayerentry.get()))
        self.testdata1=pca.fit_transform(self.yanzhen)
        self.init_Text.delete(1.0,END)
        self.init_Text.insert(1.0,self.clfmodel.predict(np.array(self.testdata1)))
        self.init_Text.insert(1.0, '产地为')

demo()
