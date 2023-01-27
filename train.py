# -*- coding： utf-8 -*-
'''
@Time: 2022/5/12 16:18
@Author:YilanZhang
@Filename:train.py
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
    epochs = options['epochs']
    modal_path = options['modal_path']
    log_path = options['log_path']
    label_t = options['labels']
    class_num = options['class_num']
    patience = options['patience']

    batch_size = options['batch_size']
    weight_decay = options['weight_decay']
    learning_rate = options['learning_rate']

    dir_release = options['dir_release']
    log_file = open(log_path + 'train_log.txt', 'w')

    #load dataset
    derm_data_group = load_dataset(dir_release=dir_release)

    # load model
    model = TFormer(class_num)

    # parallel training
    if options['cuda']:
        model = model.cuda()
    print("Model initialized")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           weight_decay=weight_decay)  # 更新网络参数，使用不同的更新规则

    # 余弦衰减
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # build a logger of training process and show it
    print('===========Training Params===============')
    log_file.write('===========Training Params===============' + '\n')
    for name, param in options.items():
        print('{}: {}'.format(name, param))
        log_file.write('{}: {}'.format(name, param) + "\n")
    print('========================================')
    log_file.write('========================================' + '\n')

    # setup training
    complete = True
    best_accuracy = 0
    train_iterator = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='train'),
                                batch_size=batch_size,shuffle=True,num_workers=4)
    valid_iterator = DataLoader(dataset(derm= derm_data_group,shape=(224,224),mode='valid'),
                                batch_size=1,shuffle=False,num_workers=2)
    test_iterator = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='test'),
                               batch_size=1,shuffle=False,num_workers=2)
    print("Start training...")
    log_file.write("Start training...")
    log_file.flush()  # 清空
    old_model_path = None
    start_time = time.time()

    try:
        for e in range(epochs):
            model.train()
            train_loss = 0.0
            print("Training epoch:{}".format(e))
            log_file.write("Training epoch:{}".format(e)+'\n')

            start_time_epoch = time.time()

            for batchIndex, (der_data,cli_data,meta_data,target) in enumerate(train_iterator):

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
                    der_data,cli_data,meta_data = der_data.cuda(), cli_data.cuda(),meta_data.cuda().float()
                der_data,cli_data,meta_data= Variable(der_data),Variable(cli_data),Variable(meta_data)

                optimizer.zero_grad()

                output = model(meta_data,cli_data,der_data)

                #multi label loss
                loss = torch.true_divide(
                    model.criterion(output[0],diagnosis_label)
                    + model.criterion(output[1],pn_label)
                    + model.criterion(output[2],bmv_label)
                    + model.criterion(output[3],vs_label)
                    + model.criterion(output[4],pig_label)
                    + model.criterion(output[5],str_label)
                    + model.criterion(output[6],dag_label)
                    + model.criterion(output[7],rs_label),8
                )

                loss.backward()
                avg_loss = loss.item()
                train_loss += avg_loss

                optimizer.step()

                if batchIndex % 50 == 0 and batchIndex > 0:
                    predicted_results = output[0].data.max(1)[1]
                    correctResultsNum = predicted_results.cpu().eq(diagnosis_label.cpu()).sum()
                    accuracy = correctResultsNum.item() * 1.0 / len(der_data)  # .item()得到一个元素张量里面的元素值
                    print('Training epoch: {} [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Learning rate: {}'.format(e,
                        batchIndex * len(der_data),len(train_iterator.dataset),loss.item(), accuracy,optimizer.param_groups[0]['lr']))

            print("Epoch {} complete! Average Training loss: {:.4f}".format(e, train_loss / len(train_iterator)))
            log_file.write("Epoch {} complete! Average Training loss: {:.4f}".format(e, train_loss / len(train_iterator)) + '\n')
            log_file.flush()

            # Terminate the training process if run into NaN
            if np.isnan(train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            # 余弦衰减
            lr_scheduler.step()

            '''Validation'''
            model.eval()
            avg_valid_loss = 0
            total_correct_results_num = 0

            val_confusion_diag = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_pn = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_bmv = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_vs = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_pig = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_str = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_dag = ConfusionMatrix(num_classes=class_num, labels=label_t)
            val_confusion_rs = ConfusionMatrix(num_classes=class_num, labels=label_t)
            for der_data, cli_data, meta_data, target in valid_iterator:
                # target = torch.squeeze(target, 1)  # torch.squeeze()对数据维数进行压缩，去掉target中维数为1的维度

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
                    der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                    der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)

                output = model(meta_data, cli_data, der_data)

                valid_loss = torch.true_divide(
                    model.criterion(output[0],diagnosis_label)
                    + model.criterion(output[1],pn_label)
                    + model.criterion(output[2],bmv_label)
                    + model.criterion(output[3],vs_label)
                    + model.criterion(output[4],pig_label)
                    + model.criterion(output[5],str_label)
                    + model.criterion(output[6],dag_label)
                    + model.criterion(output[7],rs_label),8
                )

                avg_valid_loss += valid_loss.item()

                predicted_result = output[0].data.max(1)[1]
                predictions_pn = output[1].data.max(1)[1]
                predictions_bmv = output[2].data.max(1)[1]
                predictions_vs = output[3].data.max(1)[1]
                predictions_pig = output[4].data.max(1)[1]
                predictions_str = output[5].data.max(1)[1]
                predictions_dag = output[6].data.max(1)[1]
                predictions_rs = output[7].data.max(1)[1]

                total_correct_results_num += predicted_result.cpu().eq(diagnosis_label.cpu()).sum()

                val_confusion_diag.update(int(predicted_result.cpu().numpy()),diagnosis_label.cpu().numpy())
                val_confusion_pn.update(int(predictions_pn.cpu().numpy()), pn_label.cpu().numpy())
                val_confusion_bmv.update(int(predictions_bmv.cpu().numpy()), bmv_label.cpu().numpy())
                val_confusion_vs.update(int(predictions_vs.cpu().numpy()), vs_label.cpu().numpy())
                val_confusion_pig.update(int(predictions_pig.cpu().numpy()), pig_label.cpu().numpy())
                val_confusion_str.update(int(predictions_str.cpu().numpy()), str_label.cpu().numpy())
                val_confusion_dag.update(int(predictions_dag.cpu().numpy()), dag_label.cpu().numpy())
                val_confusion_rs.update(int(predictions_rs.cpu().numpy()), rs_label.cpu().numpy())

            if np.isnan(avg_valid_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_iterator)

            dia_acc = val_confusion_diag.summary(log_file)
            pn_acc = val_confusion_pn.summary(log_file)
            bmv_acc = val_confusion_bmv.summary(log_file)
            vs_acc = val_confusion_vs.summary(log_file)
            pig_acc = val_confusion_pig.summary(log_file)
            str_acc = val_confusion_str.summary(log_file)
            dag_acc = val_confusion_dag.summary(log_file)
            rs_acc = val_confusion_rs.summary(log_file)
            accuracy_valid =100.0 * (dia_acc+pn_acc+bmv_acc+vs_acc+pig_acc+str_acc+dag_acc+rs_acc)/8.0

            print("Valid loss is:{:.4f},average accuracy:{:.4f}%".format(avg_valid_loss, accuracy_valid))
            log_file.write("Valid loss is:{:.4f},average accuracy:{:.4f}%".format(avg_valid_loss, accuracy_valid) + '\n')

            if (accuracy_valid > best_accuracy):
                curr_patience = patience
                best_accuracy = accuracy_valid
                new_model_path = os.path.join(modal_path, 'bestacc_model_{}.pth'.format(e))
                modelSnapShot(model, new_model_path, oldModelPath=old_model_path, onlyBestModel=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
            else:
                curr_patience -= 1

            if (e % 10 == 0 and e >= 30) or e==epochs:
                modelSnapShot(model, os.path.join(modal_path, 'model-' + str(e) + '.pth'))

            if curr_patience <= 0:
                break

            end_time_epoch = time.time()
            training_time_epoch = end_time_epoch - start_time_epoch
            total_training_time = time.time() - start_time
            remaining_time = training_time_epoch * epochs - total_training_time
            print("Total training time: {:.4f}s, {:.4f} s/epoch, Estimated remaining time: {:.4f}s".format(
                total_training_time, training_time_epoch, remaining_time))
        if complete:
            model.load_state_dict(torch.load(old_model_path), strict=True)

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

                confusion_diag.update(int(predictions_diag.cpu().numpy()),diagnosis_label.cpu().numpy())
                confusion_pn.update(int(predictions_pn.cpu().numpy()), pn_label.cpu().numpy())
                confusion_bmv.update(int(predictions_bmv.cpu().numpy()), bmv_label.cpu().numpy())
                confusion_vs.update(int(predictions_vs.cpu().numpy()), vs_label.cpu().numpy())
                confusion_pig.update(int(predictions_pig.cpu().numpy()), pig_label.cpu().numpy())
                confusion_str.update(int(predictions_str.cpu().numpy()), str_label.cpu().numpy())
                confusion_dag.update(int(predictions_dag.cpu().numpy()), dag_label.cpu().numpy())
                confusion_rs.update(int(predictions_rs.cpu().numpy()), rs_label.cpu().numpy())


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
OPTIONS.add_argument('--modal_path',dest='modal_path',type=str,
                     default="./result")
# log_path = options['log_path']
OPTIONS.add_argument('--log_path',dest='log_path',type=str,
                     default="./result")
# labels = options['labels']
OPTIONS.add_argument('--labels',dest='labels',default=[0,1,2,3,4])
# class_num = options['class_num']
OPTIONS.add_argument('--class_num',dest='class_num',type=int,default=5)
# patience = options['patience']
OPTIONS.add_argument('--patience',dest='patience',type=int,default=100)
# batch_size = options['batch_size']
OPTIONS.add_argument('--batch_size',dest='batch_size',type=int,default=32)
# weight_decay = options['weight_decay']
OPTIONS.add_argument('--weight_decay',dest='weight_decay',type=float,default=1e-4)
# learning_rate = options['learning_rate']
OPTIONS.add_argument('--learning_rate',dest='learning_rate',type=float,default=0.0001)

OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)

PARAMS = vars(OPTIONS.parse_args())

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = OPTIONS.parse_args()
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
