import pandas
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import os
from constant import event_type, eventType2id
from data_process.data_processor import ACE05_dataset
def get_pos_embedding_from_question():
    question_df = pandas.read_excel('./pos.xlsx')
    # print(question_df)
    definition = question_df['definition']
    # event_embedding = [np.zeros((768), dtype=float)]
    event_embedding = []
    model = BertModel.from_pretrained(r"bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(r"bert-base-uncased")
    for i in range(len(question_df)):
        # print(q)
        question_list = definition[i].replace('\n', '').split()
        # key_word_index=definition[i].replace('\n','').find(event_type[i])#key_word默认在第一个位置

        offset = {}
        embeddings = []
        question_list_tokens = [tokenizer.cls_token]
        for word_idx, word in enumerate(question_list):
            tokens = tokenizer.tokenize(word)
            for t in tokens:
                question_list_tokens.append(t)
            offset[word_idx] = len(question_list_tokens)  # 算上cls所以是len(question_list_tokens)
        question_list_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(question_list_tokens)).long()
        attnetion_mask = torch.tensor(np.ones((1, len(question_list_tokens_ids)))).long()
        segment_id = torch.tensor(np.zeros((1, len(question_list_tokens_ids)))).long()
        tokens_embeddings, _ = model(question_list_tokens_ids.view(1, -1), attnetion_mask, segment_id)
        tokens_embeddings = tokens_embeddings.squeeze(0)
        print(tokens_embeddings[offset[0]:offset[1]])
        # print("the type of embedding is ",type(tokens_embeddings))
        word_emb = torch.mean(tokens_embeddings[0:offset[0]], axis=0)
        event_embedding.append(word_emb.detach_().numpy())
    event_embedding = np.array(event_embedding)
    event_embedding = torch.from_numpy(event_embedding)
    torch.save(event_embedding, "../data/pos_vec.pkl")
def get_argument_embedding_from_question():
    question_df = pandas.read_excel('./argument.xlsx')
    # print(question_df)
    event_type = question_df['argument-type']
    definition = question_df['definition']
    event_embedding = [np.zeros((768), dtype=float)]
    # event_embedding = []
    model = BertModel.from_pretrained(r"D:\斯坦福NLP学习\pretrained")
    tokenizer = BertTokenizer.from_pretrained(r"D:\斯坦福NLP学习\pretrained")
    for i in range(len(question_df)):
        # print(q)
        question_list = definition[i].replace('\n', '').split()
        # key_word_index=definition[i].replace('\n','').find(event_type[i])#key_word默认在第一个位置

        offset = {}
        embeddings = []
        question_list_tokens = [tokenizer.cls_token]
        for word_idx, word in enumerate(question_list):
            tokens = tokenizer.tokenize(word)
            for t in tokens:
                question_list_tokens.append(t)
            offset[word_idx] = len(question_list_tokens)  # 算上cls所以是len(question_list_tokens)
        question_list_tokens.append(tokenizer.sep_token)
        question_list_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(question_list_tokens)).long()
        attnetion_mask = torch.tensor(np.ones((1, len(question_list_tokens_ids)))).long()
        segment_id = torch.tensor(np.zeros((1, len(question_list_tokens_ids)))).long()
        tokens_embeddings, _ = model(question_list_tokens_ids.view(1, -1), attnetion_mask, segment_id)
        tokens_embeddings = tokens_embeddings.squeeze(0)
        print(tokens_embeddings[offset[0]:offset[1]])
        # print("the type of embedding is ",type(tokens_embeddings))
        word_emb = torch.mean(tokens_embeddings[1:offset[0]], axis=0)
        event_embedding.append(word_emb.detach_().numpy())
    event_embedding = np.array(event_embedding)
    event_embedding = torch.from_numpy(event_embedding)
    torch.save(event_embedding, "../data/argument_vec.pkl")
def get_entity_embedding_from_question():
    question_df = pandas.read_excel('./entity.xlsx')
    # print(question_df)
    event_type = question_df['entity-type']
    definition = question_df['definition']
    event_embedding = [np.zeros((768), dtype=float)]
    # event_embedding = []
    model = BertModel.from_pretrained(r"bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(r"bert-base-uncased")
    for i in range(len(question_df)):
        # print(q)
        question_list = definition[i].replace('\n', '').split()
        # key_word_index=definition[i].replace('\n','').find(event_type[i])#key_word默认在第一个位置

        offset = {}
        embeddings = []
        question_list_tokens = [tokenizer.cls_token]
        for word_idx, word in enumerate(question_list):
            tokens = tokenizer.tokenize(word)
            for t in tokens:
                question_list_tokens.append(t)
            offset[word_idx] = len(question_list_tokens)  # 算上cls所以是len(question_list_tokens)
        question_list_tokens.append(tokenizer.sep_token)
        question_list_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(question_list_tokens)).long()
        attnetion_mask = torch.tensor(np.ones((1, len(question_list_tokens_ids)))).long()
        segment_id = torch.tensor(np.zeros((1, len(question_list_tokens_ids)))).long()
        tokens_embeddings, _ = model(question_list_tokens_ids.view(1, -1), attnetion_mask, segment_id)
        tokens_embeddings = tokens_embeddings.squeeze(0)
        print(tokens_embeddings[offset[0]:offset[1]])
        # print("the type of embedding is ",type(tokens_embeddings))
        word_emb = torch.mean(tokens_embeddings[1:offset[0]], axis=0)
        event_embedding.append(word_emb.detach_().numpy())
    event_embedding = np.array(event_embedding)
    event_embedding = torch.from_numpy(event_embedding)
    torch.save(event_embedding, "../data/base/entity_vec.pkl")
def get_event_embedding_from_question():
    question_df = pandas.read_excel('./question.xlsx')
    # print(question_df)
    event_type = question_df['event-type']
    definition = question_df['definition']
    event_embedding = [np.zeros((768), dtype=float)]
    # event_embedding = []
    model = BertModel.from_pretrained(r"bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(r"bert-base-uncased")
    for i in range(1,len(question_df)):
        # print(q)
        question_list = definition[i].replace('\n', '').split()
        # key_word_index=definition[i].replace('\n','').find(event_type[i])#key_word默认在第一个位置

        offset = {}
        embeddings = []
        question_list_tokens = [tokenizer.cls_token]
        for word_idx, word in enumerate(question_list):
            tokens = tokenizer.tokenize(word)
            for t in tokens:
                question_list_tokens.append(t)
            offset[word_idx] = len(question_list_tokens)  # 算上cls所以是len(question_list_tokens)
        question_list_tokens.append(tokenizer.sep_token)
        question_list_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(question_list_tokens)).long()
        attnetion_mask = torch.tensor(np.ones((1, len(question_list_tokens_ids)))).long()
        segment_id = torch.tensor(np.zeros((1, len(question_list_tokens_ids)))).long()
        tokens_embeddings, cls_embedding = model(question_list_tokens_ids.view(1, -1), attnetion_mask, segment_id)
        tokens_embeddings = tokens_embeddings.squeeze(0)
        print(tokens_embeddings[offset[0]:offset[1]])
        # print("the type of embedding is ",type(tokens_embeddings))
        word_emb = torch.mean(tokens_embeddings[1:offset[0]], axis=0)
        event_embedding.append(word_emb.detach_().numpy())
    event_embedding = np.array(event_embedding)
    event_embedding = torch.from_numpy(event_embedding)
    torch.save(event_embedding, "../data/base/event_vec_from_question.pkl")

def get_event_embedding_from_word():
    question_df = pandas.read_excel('./question.xlsx')
    # print(question_df)
    event_type = question_df['event-type']
    definition = event_type
    event_embedding = [np.zeros((768), dtype=float)]
    model = BertModel.from_pretrained(r"D:\实验室\NER\BERT_word_embedding\pretrained")
    tokenizer = BertTokenizer.from_pretrained(r"D:\实验室\NER\BERT_word_embedding\pretrained")
    for i in range(len(question_df)):
        question = definition[i]
        question_list_tokens = [tokenizer.cls_token]+tokenizer.tokenize(question)+[tokenizer.sep_token]
        question_list_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(question_list_tokens)).long()
        attnetion_mask = torch.tensor(np.ones((1, len(question_list_tokens_ids)))).long()
        segment_id = torch.tensor(np.zeros((1, len(question_list_tokens_ids)))).long()
        _, cls_embedding = model(question_list_tokens_ids.view(1, -1), attnetion_mask, segment_id)
        event_embedding.append(cls_embedding.squeeze(0).detach().numpy())
    event_embedding = np.array(event_embedding)
    event_embedding = torch.from_numpy(event_embedding)
    torch.save(event_embedding, "../data/event_vec_from_word.pkl")

def get_event_embedding_from_instance(processor):
    bert = BertModel.from_pretrained(r'bert-base-uncased').to('cuda')
    cached_examples_file = os.path.join("../data/cached_train")
    if os.path.exists(cached_examples_file):
        instances = torch.load(cached_examples_file)
    else:
        instances = processor.get_train_examples()
    result = []
    type2token = {i: [] for i in range(len(event_type))}
    type2attention = {i: [] for i in range(len(event_type))}
    for single_instance in instances:
        type = single_instance['event_type']
        type_id = eventType2id[type]
        type2token[type_id].append(single_instance['tokens_id'])
        type2attention[type_id].append(single_instance['attention'])
    for i in range(len(event_type)):
        tmp = []
        tokens_id_list = type2token[i]
        attention_list = type2attention[i]
        for num, id in enumerate(tokens_id_list):
            tokens_id = torch.LongTensor(np.array(id)).to('cuda')
            attention_mask = torch.LongTensor(np.array(attention_list[num])).to('cuda')
            cls = bert(tokens_id.view(1, len(id)), attention_mask.view(1, len(id)))[1].squeeze(0).cpu().detach().numpy()
            tmp.append(cls)
        if len(tmp)==0:
            tmp = np.random.randn(768)
        else:
            tmp = np.array(tmp)
            tmp = np.mean(tmp, axis=0)  # 所有同类事件cls求平均
        result.append(tmp)
    argument_vec = np.array(result)
    argument_vec = torch.from_numpy(argument_vec)
    torch.save(argument_vec, "../data/event_vec_from_word.pkl")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    get_argument_embedding_from_question()
    print(1)
    # if not os.path.exists("../data/event_vec_from_question.pkl"):
    #     get_entity_embedding_from_question()
    # tmp = torch.load('../data/base/event_vec_from_question.pkl')
    # embedding = torch.nn.Embedding.from_pretrained(tmp)
    # for i in range(34):
    #     for j in range(34):
    #         embedding1 = embedding(torch.scalar_tensor(i).long()).view(1, -1)
    #         embedding2 = embedding(torch.scalar_tensor(j).long()).view(1, -1)
    #         embedding1 = F.normalize(embedding1)
    #         embedding2 = F.normalize(embedding2)
    #         distance = embedding1.mm(embedding2.t()).data
    #         print("argument role: {} ----{} cosine_similarity is {}".format(event_type[i],
    #                                                                         event_type[j],
    #                                                                         distance))
