# -*- coding： utf-8 -*-
'''
@Time: 2024/1/29 17:07
@Author:YilanZhang
@Filename:test.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import time
import torch.nn.functional as F

from src.eval_metrics import ConfusionMatrix
from models.TFormer import TFormer
from src.dataloader import load_dataset,dataset,train_data_transformation,test_data_transformation

'''function for saving model'''
def modelSnapShot(model,newModelPath,oldModelPath=None,onlyBestModel=False):
    if onlyBestModel and oldModelPath:
        os.remove(oldModelPath)
    torch.save(model.state_dict(),newModelPath)


def main(options):
    # parse the input args
    model_path = options['model_path']
    log_path = options['log_path']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    label_t = options['labels']
    class_num = options['class_num']


    dir_release = options['dir_release']
    log_file = open(log_path + 'test_log.txt', 'w')

    #load dataset
    derm_data_group = load_dataset(dir_release=dir_release)

    # load model
    model = TFormer(class_num)

    # parallel training
    if options['cuda']:
        model = model.cuda()
    print("Model initialized")

    test_iterator = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='test'),
                               batch_size=1,shuffle=False,num_workers=2)
    print("Start testing...")
    log_file.write("Start testing...")
    log_file.flush()

    try:

        model.load_state_dict(torch.load(model_path), strict=True)

        model.eval()

        avg_test_loss = 0
        confusion_diag =ConfusionMatrix(num_classes=class_num,labels =label_t)
        confusion_pn = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_bmv = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_vs = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_pig = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_str = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_dag = ConfusionMatrix(num_classes=class_num, labels=label_t)
        confusion_rs = ConfusionMatrix(num_classes=class_num, labels=label_t)

        for der_data,cli_data,meta_data,target in test_iterator:
            # target=torch.squeeze(target,1) #torch.squeeze()对数据维数进行压缩，去掉target中维数为1的维度
            # Diagostic label
            diagnosis_label = target[0].squeeze(1).cuda()
            # Seven-Point Checklikst labels
            pn_label = target[1].squeeze(1).cuda()
            bmv_label = target[2].squeeze(1).cuda()
            vs_label = target[3].squeeze(1).cuda()
            pig_label = target[4].squeeze(1).cuda()
            str_label = target[5].squeeze(1).cuda()
            dag_label = target[6].squeeze(1).cuda()
            rs_label = target[7].squeeze(1).cuda()
            if options['cuda']:
                der_data,cli_data,meta_data= der_data.cuda(), cli_data.cuda(),meta_data.cuda().float()
                der_data,cli_data,meta_data= Variable(der_data),Variable(cli_data),Variable(meta_data)

            output = model(meta_data,cli_data,der_data)

            test_loss = torch.true_divide(
                model.criterion(output[0],diagnosis_label)
                + model.criterion(output[1],pn_label)
                + model.criterion(output[2],bmv_label)
                + model.criterion(output[3],vs_label)
                + model.criterion(output[4],pig_label)
                + model.criterion(output[5],str_label)
                + model.criterion(output[6],dag_label)
                + model.criterion(output[7],rs_label),8
            )
            avg_test_loss += test_loss.item()

            #confusion matrix
            ret,predictions_diag = torch.max(output[0].data,1)
            ret, predictions_pn = torch.max(output[1].data, 1)
            ret, predictions_bmv = torch.max(output[2].data, 1)
            ret, predictions_vs = torch.max(output[3].data, 1)
            ret, predictions_pig = torch.max(output[4].data, 1)
            ret, predictions_str = torch.max(output[5].data, 1)
            ret, predictions_dag = torch.max(output[6].data, 1)
            ret, predictions_rs = torch.max(output[7].data, 1)

            confusion_diag.update(predictions_diag.cpu().numpy(),diagnosis_label.cpu().numpy())
            confusion_pn.update(predictions_pn.cpu().numpy(), pn_label.cpu().numpy())
            confusion_bmv.update(predictions_bmv.cpu().numpy(), bmv_label.cpu().numpy())
            confusion_vs.update(predictions_vs.cpu().numpy(), vs_label.cpu().numpy())
            confusion_pig.update(predictions_pig.cpu().numpy(), pig_label.cpu().numpy())
            confusion_str.update(predictions_str.cpu().numpy(), str_label.cpu().numpy())
            confusion_dag.update(predictions_dag.cpu().numpy(), dag_label.cpu().numpy())
            confusion_rs.update(predictions_rs.cpu().numpy(), rs_label.cpu().numpy())


        print("Daig:\n")
        log_file.write("Daig:\n")
        confusion_diag.summary(log_file)
        print("PN:\n")
        log_file.write("PN:\n")
        confusion_pn.summary(log_file)
        print("BMV:\n")
        log_file.write("BMV:\n")
        confusion_bmv.summary(log_file)
        print("VS:\n")
        log_file.write("VS:\n")
        confusion_vs.summary(log_file)
        print("PIG:\n")
        log_file.write("PIG:\n")
        confusion_pig.summary(log_file)
        print("STR:\n")
        log_file.write("STR:\n")
        confusion_str.summary(log_file)
        print("DAG:\n")
        log_file.write("DAG:\n")
        confusion_dag.summary(log_file)
        print("RS:\n")
        log_file.write("RS:\n")
        confusion_rs.summary(log_file)
        log_file.flush()

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        log_file.close()


OPTIONS = argparse.ArgumentParser()

# # parse the input args
# epochs = options['epochs']
OPTIONS.add_argument('--epochs',dest='epochs',type=int,default=100)
# dir_release = options['dir_release']
OPTIONS.add_argument('--dir_release',dest='dir_release',type=str,default="./data/derm7pt/release_v0/")
# modal_path = options['modal_path']
# Download the pretrained model from https://huggingface.co/vemvet/TFormer/tree/main
OPTIONS.add_argument('--model_path',dest='model_path',type=str, default="./result/new_model.pth")
# log_path = options['log_path']
OPTIONS.add_argument('--log_path',dest='log_path',type=str,
                     default="./result/")
# labels = options['labels']
OPTIONS.add_argument('--labels',dest='labels',default=[0,1,2,3,4])
# class_num = options['class_num']
OPTIONS.add_argument('--class_num',dest='class_num',type=int,default=5)

OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)

PARAMS = vars(OPTIONS.parse_args())

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = OPTIONS.parse_args()
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
