from copy import deepcopy
from random import randint
import xlrd
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader

# 打乱 list 中句子的顺序
def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
      m -= 1
      i = randint(0, m)
      temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst


def main():
    f = xlrd.open_workbook('Ko2Cn.xlsx').sheet_by_name('Xbench QA')
    Ko_list = f.col_values(0) #　所有的中文句子
    Cn_list = f.col_values(1) #　所有的韩语句子

    shuffle_Cn_list = shuffle(Cn_list) # 所有的中文句子打乱排序
    shuffle_Ko_list = shuffle(Ko_list) #　所有的韩语句子打乱排序

    train_size = int(len(Ko_list) * 0.8)
    eval_size = len(Ko_list) - train_size

    # Define your train examples.
    train_data = []
    for idx in range(train_size):
      train_data.append(InputExample(texts=[Ko_list[idx], Cn_list[idx]], label=1.0))
      train_data.append(InputExample(texts=[shuffle_Ko_list[idx], shuffle_Cn_list[idx]], label=0.0))

    # Define your evaluation examples
    sentences1 = Ko_list[train_size:]
    sentences2 = Cn_list[train_size:]
    sentences1.extend(list(shuffle_Ko_list[train_size:]))
    sentences2.extend(list(shuffle_Cn_list[train_size:]))
    scores = [1.0] * eval_size + [0.0] * eval_size

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    # Define your train dataset, the dataloader and the train loss
    train_dataset = SentencesDataset(train_data, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer('distiluse-base-multilingual-cased')

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator,
              evaluation_steps=100, output_path='./PoemModel')


if __name__ == '__main__':
    main()


