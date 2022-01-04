import numpy as np
import json
import os
import torch
from transformers import BertTokenizer
from data_process.constant import event_type,entityType2id,pos_tag2id,pos2type,eventType2id,all_arguments,argument2id
import copy
class ACE05_dataset():
    def __init__(self,filepath,tokenizer:BertTokenizer,max_length):
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.filepath=filepath

    def get_train_examples(self):
        """See base class."""
        with open(os.path.join(self.filepath,"train.json")) as infile:
            rawdata = json.load(infile)  # 训练数据集
        return self.process(rawdata)#

    def get_test_examples(self):
        with open(os.path.join(self.filepath,"test.json")) as infile:
            rawdata = json.load(infile)  # test数据集
        return self.process(rawdata)

    def get_dev_examples(self):
        with open(os.path.join(self.filepath,"dev.json")) as infile:
            rawdata = json.load(infile)  # dev数据集
        return self.process(rawdata)
    def sort_entity(self,entity_mention):
        #按照span从小到大排列
        entity_mention = sorted(entity_mention, key=lambda x: x['end']-x['start'], reverse=False)
        return entity_mention

    def process(self,raw_data):
        example=[]
        for single_json in raw_data:

            words=single_json['words']
            entity_mention=single_json['golden-entity-mentions']
            entity_mention=self.sort_entity(entity_mention)
            event_mention=single_json['golden-event-mentions']
            tokens=[]
            offset=[]
            for i,w in enumerate(words):
                tokens.extend(self.tokenizer.tokenize(w))
                offset.extend([i+1]*len(self.tokenizer.tokenize(w)))#+1为了考虑cls
            tmp_offset=[offset[0]]
            for i in range(1,len(offset)):
                if offset[i]==offset[i-1]:
                    tmp_offset.append(0)
                else:
                    tmp_offset.append(offset[i])
            offset_mask=np.array(tmp_offset)!=0
            tokens=list(np.array(tokens)[offset_mask])
            assert  len(tokens)==len(words)
            # tokens = self.tokenizer.tokenize(sentence)#原始token，没有cls sep
            padding_length = self.max_length - len(words)-2
            '''实体标签'''
            entity_id=[0]+self.entity_extractor(entity_mention,words)+[0]
            if len(entity_id)==2:
                continue
            '''事件标签'''
            padding_block = np.array([0] * len(all_arguments))  # padding是一个矩阵不是一个数字。
            padding_block[0] = 1
            if event_mention!=[]:
                event_type_list=[]
                arguments=[]
                for event in event_mention:
                    arguments.extend(event['arguments'])
                argument_id = self.argument_and_event_extractor(arguments, words)
                if len(argument_id[0]) == 20:
                    print(1)
                for i,event in enumerate(event_mention):
                    event_type_list.append(eventType2id[event['event_type'].split(":")[1]])

                event_type_id=np.zeros(len(event_type))
                event_type_id[np.array(event_type_list)]=1
                tmp_tokens=tokens
                entity_id = [0] + self.entity_extractor(entity_mention, words) + [0]
                if len(entity_id)==2:
                    continue

                json_d = {}

                if padding_length >= 0:
                    tmp_tokens = [self.tokenizer.cls_token] + tmp_tokens + [self.tokenizer.sep_token]
                    attention = [1] * len(tmp_tokens) + [0] * padding_length
                    tmp_tokens.extend(padding_length * [self.tokenizer.pad_token])
                    entity_id.extend([0] * padding_length)
                    length = self.max_length - padding_length
                    if padding_length!=0:
                        padding_array=copy.deepcopy(padding_block)
                        for i in range(padding_length-1):
                            padding_array=np.vstack((padding_array,padding_block))

                        argument_id=np.vstack((argument_id,padding_array))

                if padding_length < 0:
                    tmp_tokens = [self.tokenizer.cls_token] + tmp_tokens
                    tmp_tokens = tmp_tokens[:self.max_length]
                    tmp_tokens[-1] = self.tokenizer.sep_token
                    attention = [1] * self.max_length
                    entity_id = entity_id[:self.max_length]
                    entity_id[-1] = 0
                    length = len(words)+2
                    argument_id = argument_id[:self.max_length]
                    argument_id[-1,:] = padding_block
                assert len(tmp_tokens) == len(entity_id) == len(argument_id) == len(attention)
                tokens_id = self.tokenizer.convert_tokens_to_ids(tmp_tokens)
                '''验证一下各个部分是否正确'''
                json_d['event_type']=event_type_id
                json_d['length'] = length
                json_d["tokens"]=tmp_tokens
                json_d['tokens_id']=tokens_id
                json_d['attention']=attention
                json_d['entity_id']=entity_id
                json_d['argument_id']=argument_id
                example.append(json_d)
            else:
                event_type_id=np.zeros(len(event_type))

                event_type_id[0]=1
                tmp_tokens = tokens
                argument_id=copy.deepcopy(padding_block)
                for i in range(self.max_length-1):
                    argument_id = np.vstack((argument_id, padding_block))

                if padding_length >= 0:
                    tmp_tokens = [self.tokenizer.cls_token] + tmp_tokens + [self.tokenizer.sep_token]
                    attention = [1] * len(tmp_tokens) + [0] * padding_length
                    tmp_tokens.extend(padding_length * [self.tokenizer.pad_token])
                    entity_id.extend([0] * padding_length)
                    length = self.max_length - padding_length

                if padding_length < 0:
                    tmp_tokens = [self.tokenizer.cls_token] + tmp_tokens
                    tmp_tokens = tmp_tokens[:self.max_length]
                    tmp_tokens[-1] = self.tokenizer.sep_token
                    attention = [1] * self.max_length
                    entity_id = entity_id[:self.max_length]
                    entity_id[-1] = 0
                    length = len(words)+2

                assert len(tmp_tokens) == len(entity_id) == len(argument_id) == len(attention)

                tokens_id = self.tokenizer.convert_tokens_to_ids(tmp_tokens)
                json_d={}
                json_d['event_type'] = event_type_id
                json_d['length'] = length
                json_d["tokens"] = tmp_tokens
                json_d['tokens_id'] = tokens_id
                json_d['attention'] = attention
                json_d['entity_id'] = entity_id
                json_d['argument_id'] = argument_id
                example.append(json_d)
        return example

    def entity_extractor(self,entity_mentions,words):
        entity_type_start_end_list = []
        if entity_mentions==[]:
            return [0]*len(words)
        for mention in entity_mentions:
            text = mention['text']
            entity_type = mention['entity-type']
            raw_start = mention['start']  # [start,end)左闭右开
            raw_end = mention['end']  # [start,end)左闭右开)
            error_flag=True
            for word in words[raw_start:raw_end]:
                if word in text or text in word:
                    error_flag=False
            if error_flag:
                # TODO（高天昊）这里有些实例会报玄学错误，不过多数是json数据错误。以后有空修复，目前大部分实例都算正常
                print("entity error return all zero")
                return [0]*len(words)
            entity_type_start_end_list.append((entity_type, raw_start, raw_end))
        entity_id = self.get_tags_id(words, entity_type_start_end_list)  # 得到实体的ids,两侧补的0是cls和sep
        return entity_id
    def argument_and_event_extractor(self,mention,words):
        role_start_end_list = []
        if mention==[]:#没事件，返回全0
            padding_block = np.array([0] * len(all_arguments))  # padding是一个矩阵不是一个数字。
            padding_block[0] = 1
            padding_array=copy.deepcopy(padding_block)
            for i in range(len(words)-1+2):#2是cls和sep
                padding_array=np.vstack((padding_array,padding_block))
            return padding_array
        argument_list = mention
        res_list=[]
        for argument in argument_list:
            text=argument["text"]
            text_tokens = self.tokenizer.tokenize(text)  # 重新确定开始结束位置。
            raw_start = argument['start']  # [start,end)左闭右开
            raw_end = argument['end']
            error_flag=True
            for word in words[raw_start:raw_end]:
                if word in text or text in word:
                    error_flag = False
            if error_flag:
                # TODO（高天昊）这里有些实例会报玄学错误，不过多数是json数据错误。以后有空修复，目前大部分实例都算正常
                print("raw argument error,skip one argument")
                return [0]*len(words)

            argument_id = self.get_tags_id(words, [[argument['role'], raw_start, raw_end]], "argument")
            tmp_argument_id = [0] + argument_id + [0]
            tmp_argument_id = torch.nn.functional.one_hot(torch.tensor(tmp_argument_id), len(all_arguments)).numpy()
            res_list.append(tmp_argument_id)
        argument_id = np.sum(np.array(res_list), axis=0)
        argument_id[:,0]=np.where(argument_id[:,0]==len(res_list),1,0)
        argument_id[:, 1:] = np.where(argument_id[:, 1:] !=0, 1, 0)
        return argument_id


    def pos_extractor(self,tokens,single_json,offset):
        pos_tags=single_json['pos-tags']
        tokens_pos_tags=np.array([0]*len(tokens))
        word_nums=len(single_json['words'])
        assert (word_nums-1)==offset[-1]
        for i in range(word_nums):
            mask=np.where(np.array(offset)==i)
            tokens_pos_tags[mask]=pos_tag2id[pos2type[pos_tags[i]]]
        return tokens_pos_tags.tolist()

    def get_tags_id(self,words,type_start_end_list,type="entity"):
        if type not in ["entity","argument"]:
            raise ValueError("the type should entity or argument")
        if type=='entity':
            tag=[0]*len(words)
            for entity in type_start_end_list:
                entity_type,start,end = entity
                entity_type=entity_type.split(":")[0]#粗糙，如果是fine那么就删掉
                for position in range(start,end):
                    tag[position] = entityType2id[entity_type]
            return tag
        if type =="argument":
            BIO = [0] * len(words)
            for argumenmts in type_start_end_list:
                role, start, end = argumenmts
                for position in range(start, end):
                    BIO[position] = argument2id[role]
            return BIO
        # if type =="argument":
        #     BIO = [0] * len(tokens)
        #     for argumenmts in type_start_end_list:
        #         role, start, end = argumenmts
        #         role_type=all_arguments[role]
        #         for position in range(start, end):
        #             if position == start:
        #                 BIO[position] = argument2id["B-" + role_type]
        #             else:
        #                 BIO[position] = argument2id["I-" + role_type]
        #     return BIO

