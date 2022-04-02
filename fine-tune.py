import csv
import random
from copy import deepcopy
from random import randint
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import os


# 打乱 list 中句子的顺序
def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while m:
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

def main():
    model_save_path = 'model'
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    base = './res/data'
    file_list = []
    for i in findAllFile(base):
        print(i)
        file_list.append(i)
    print("file nums: " + str(len(file_list)))

    # 主题 译文 赏析
    allzip = []

    for eachfile in file_list:
        readfile = open('./res/data/' + eachfile, 'r', encoding='utf-8')
        reader = csv.reader(readfile)
        # ['标题', '作者', '原文句子', '译文句子', '赏析', '原文分词', '译文分词', '赏析分词']
        row = []
        # 遍历这个文件的每一行
        for e in reader:
            if len(e) > 0:
                row.append(e)
                # eachfile, yiwen, shangxi 构造训练集
                allzip.append([eachfile[:-4]] + [e[3]] + [e[4]])
        # 空文件过掉
        if len(row) == 0:
            continue

    print("all available datas: " + str(len(allzip)))

    print("filename    yiwen    shangxi")
    print(list(allzip)[0])

    # 构造负样本
    filename_list = [row[:-4] for row in file_list]
    yiwen_list = [row[1] for row in allzip]
    shangxi_list = [row[2] for row in allzip]
    print(filename_list)
    print('----------------------')
    print(yiwen_list[5])
    print('----------------------')
    print(shangxi_list[5])


    shuffle_filename = shuffle(filename_list)
    shuffle_yiwen = shuffle(yiwen_list)
    shuffle_shangxi = shuffle(shangxi_list)

    train_size = int(len(allzip) * 0.7)
    eval_size = int(len(allzip) * 0.2)
    test_size = int(len(allzip) * 0.1)

    train_data = []
    filename_i = random.randint(0, len(shuffle_filename) - 1)
    yiwen_i = random.randint(0, len(shuffle_yiwen) - 1)

    # 正样本
    print("add positive train data")
    for idx in range(train_size):
        train_data.append(InputExample(texts=[allzip[idx][0], allzip[idx][1]], label=1.0))
        train_data.append(InputExample(texts=[shuffle_filename[filename_i], shuffle_yiwen[yiwen_i]], label=0.0))
        filename_i = random.randint(0, len(shuffle_filename) - 1)
        yiwen_i = random.randint(0, len(shuffle_yiwen) - 1)

    # define evaluation examples
    print("define evaluation examples")

    sentences1 = [row[0] for row in allzip[train_size: train_size + eval_size]]
    print('sentences1: ')
    print(sentences1)

    sentences2 = [row[1] for row in allzip[train_size: train_size + eval_size]]
    print("yiwen: ")
    print(sentences2)

    filename_i = random.randint(0, len(shuffle_filename) - 1)
    yiwen_i = random.randint(0, len(shuffle_yiwen) - 1)

    for e in range(eval_size):
        sentences1.append(shuffle_filename[filename_i])
        sentences2.append(shuffle_yiwen[yiwen_i])
        filename_i = random.randint(0, len(shuffle_filename) - 1)
        yiwen_i = random.randint(0, len(shuffle_yiwen) - 1)

    print("evaluation query: ")
    print(sentences1)

    scores = [1.0] * eval_size + [0.0] * eval_size

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    # Define your train dataset, the dataloader and the train loss
    train_dataset = SentencesDataset(train_data, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator,
              evaluation_steps=100, output_path=model_save_path)



if __name__ == '__main__':
    main()
