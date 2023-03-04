# -*- coding： utf-8 -*-
'''
@Time: 2022/5/12 16:09
@Author:YilanZhang
@Filename:eval_metrics.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
import torch


'''Confusion Matrix'''
class ConfusionMatrix(object):

    def __init__(self,num_classes:int,labels:list):
        self.matrix=np.zeros((num_classes,num_classes)) #initial matrix
        self.num_classes=num_classes # classes num
        self.labels=labels # classes labels
        self.PrecisionofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SensitivityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SpecificityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.acc = 0.0


    def update(self,pred,label):
        if len(pred)>1:
            for p,t in zip(pred, label): 
                self.matrix[int(p),int(t)] += 1 
        else:
            self.matrix[int(pred),int(label)] += 1

    def summary(self,File):
        #calculate accuracy
        sum_TP=0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i] #混淆矩阵对角线的元素之和，也就是分类正确的数量
        self.acc = sum_TP/np.sum(self.matrix) #总体准确率
        print("the model accuracy is ",self.acc)
        File.write("the model accuracy is {}".format(self.acc)+"\n")

        #precision,recall,specificity
        table=PrettyTable() # create a table
        table.field_names=["","Precision","Sensitivity","Specificity"]
        for i in range(self.num_classes):
            TP=self.matrix[i,i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision=round(TP/(TP+FP),4) if TP+FP!=0 else 0.
            Sensitivity=round(TP/(TP+FN),4) if TP+FN!=0 else 0.
            Specificity=round(TN/(TN+FP),4) if TN+FP!=0 else 0.

            self.PrecisionofEachClass[i]=Precision
            self.SensitivityofEachClass[i]=Sensitivity
            self.SpecificityofEachClass[i]=Specificity

            table.add_row([self.labels[i],Precision,Sensitivity,Specificity])
        print(table)
        File.write(str(table)+'\n')
        return self.acc

    def plot(self):#plot matrix
        matrix=self.matrix
        print(matrix)
        plt.imshow(matrix,cmap=plt.cm.Blues)

        #x label
        plt.xticks(range(self.num_classes),self.labels,rotation=45)
        #y label
        plt.yticks(range(self.num_classes),self.labels)
        #show colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix (acc='+self.summary()+')')

       
        thresh=matrix.max()/2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info=int(matrix[y,x])
                plt.text(x,y,info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "block")

        plt.tight_layout()
        plt.show()


'''ROC AUC'''
def calculate_auc(pro_list,lab_list,classnum,File):
    pro_array = np.array(pro_list)
    #label to onehot
    lab_tensor = torch.tensor(lab_list)
    lab_tensor = lab_tensor.reshape((lab_tensor.shape[0],1))
    lab_onehot = torch.zeros(lab_tensor.shape[0],classnum)
    lab_onehot.scatter_(dim=1, index=lab_tensor, value=1)
    lab_onehot = np.array(lab_onehot)

    table = PrettyTable()
    table.field_names = ["", "auc"]
    roc_auc = []
    for i in range(classnum):
        fpr,tpr,_=roc_curve(lab_onehot[:,i],pro_array[:,i])
        auc_i=auc(fpr, tpr)
        roc_auc.append(auc_i)
        table.add_row([i,auc_i])
    print(table)
    File.write(str(table) + '\n')
    print(np.mean(roc_auc))
    # return np.mean(roc_auc)




