import torch
import torch.nn.functional as F
import os
import math
from transformers import BertTokenizer, BertModel
import numpy as np
from constant import all_arguments, argument2id, id2argument,event_type
from data_process.data_processor import ACE05_dataset

model = BertModel.from_pretrained(r"D:\实验室\NER\BERT_word_embedding\pretrained")
tokenizer = BertTokenizer.from_pretrained(r"D:\实验室\NER\BERT_word_embedding\pretrained")

def get_argument_embedding(processor, argument_list):
    result = []
    instances = processor.get_train_examples()
    for current_id, argument_type in enumerate(argument_list):
        embedding_list = []
        for single_instance in instances:
            argument_id = single_instance['argument_id']
            argument_mask = (np.array(argument_id) == current_id)
            argument_mask_binary = argument_mask + 0
            tokens_id = torch.tensor(single_instance['tokens_id']).long().view(1, -1)
            '''
                attention_mask=torch.tensor(argument_mask+0).long()#实现方法一
            '''
            attention_mask = torch.ones((1, tokens_id.shape[1])).long()  # 实现方法二
            segment_id = torch.tensor(np.zeros((1, tokens_id.shape[1]))).long()
            embedding = model(tokens_id, attention_mask, segment_id)[0].detach_()
            if np.sum(argument_mask_binary) == 0:
                embedding_list.append(np.zeros(shape=(768,)))
            else:
                wordpiece_embedding = embedding.squeeze(0).numpy()
                mean_embedding = np.mean(wordpiece_embedding[argument_mask], axis=0)
                embedding_list.append(mean_embedding)
            print(current_id)
        result.append(np.mean(np.array(embedding_list), axis=0))
    argument_vec = np.array(result)
    argument_vec = torch.from_numpy(argument_vec)
    torch.save(argument_vec, "../data/argument_vec.pkl")
def get_argument_embedding2(processor, argument_list):
    result = []
    instances = processor.get_train_examples()[:2000]
    for current_id, argument_type in enumerate(argument_list):
        embedding_list = []
        num=0
        for single_instance in instances:
            argument_id = single_instance['argument_id']
            argument_mask = (np.array(argument_id) == current_id)
            argument_mask[0]=1
            argument_mask_binary = argument_mask + 0
            tokens_id = torch.tensor(single_instance['tokens_id']).long().view(1, -1)
            attention_mask=torch.tensor(argument_mask).long().view(1,-1)#实现方法一
            '''
            # attention_mask = torch.ones((1, tokens_id.shape[1])).long()  # 实现方法二
            '''
            segment_id = torch.tensor(np.zeros((1, tokens_id.shape[1]))).long()
            embedding = model(tokens_id, attention_mask, segment_id)[1].detach_()
            if np.sum(argument_mask_binary) == 0:
                continue
            else:
                cls = embedding.squeeze(0).numpy()
                embedding_list.append(cls)
            print(current_id)
        result.append(np.mean(np.array(embedding_list), axis=0))
    argument_vec = np.array(result)
    argument_vec = torch.from_numpy(argument_vec)
    torch.save(argument_vec, "../data/argument_vec2.pkl")


# def cosine_similarity(embedding1, embedding2, dim=768):
#     return sum(embedding2 * embedding1) / (math.sqrt(sum(embedding1 ^ 2)) * math.sqrt(sum(embedding2 ^ 2)))


if __name__ == "__main__":
    argument_list = all_arguments
    processor = ACE05_dataset("../data", tokenizer, 32)
    if not os.path.exists("../data/argument_vec.pkl"):
        get_argument_embedding(processor, argument_list)
    tmp = torch.load('../data/argument_vec.pkl')
    embedding = torch.nn.Embedding(37, 768).from_pretrained(tmp)
    print(embedding(torch.scalar_tensor(6).long()).view(1, -1))
    for i in range(37):
        for j in range(37):
            embedding1 = embedding(torch.scalar_tensor(i).long()).view(1,-1)
            embedding2 = embedding(torch.scalar_tensor(j).long()).view(1,-1)
            embedding1=F.normalize(embedding1)
            embedding2=F.normalize(embedding2)
            distance = embedding1.mm(embedding2.t()).data
            print("argument role: {} ----{} cosine_similarity is {}".format(all_arguments[i],
                                                                                        all_arguments[j],
                                                                                       distance))
