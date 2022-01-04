import numpy as np
import json
import os
import torch
from transformers import BertTokenizer
from constant import entityType2id,argument2id,pos_tag2id,pos2type


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

    def process(self,raw_data):
        example=[]
        for single_json in raw_data:
            words=single_json['words']
            entity_mention=single_json['golden-entity-mentions']
            event_mention=single_json['golden-event-mentions']
            # if len(event_mention)==0:#如果这个instance没有event
            #     continue
            tokens=[]
            offset=[]
            for i,w in enumerate(words):
                tokens.extend(self.tokenizer.tokenize(w))
                offset.extend([i]*len(self.tokenizer.tokenize(w)))
            # tokens = self.tokenizer.tokenize(sentence)#原始token，没有cls sep
            padding_length = self.max_length - len(tokens)-2
            '''实体标签'''
            entity_id=[0]+self.entity_extractor(tokens,entity_mention,words)+[0]
            if len(entity_id)==2:
                continue
            '''词性标签'''
            pos_id=[0]+self.pos_extractor(tokens,single_json,offset)+[0]
            print(pos_id)
            '''事件标签'''

            if event_mention!=[]:
                for event in event_mention:
                    tokens=[]
                    for i, w in enumerate(words):
                        tokens.extend(self.tokenizer.tokenize(w))
                    entity_id = [0] + self.entity_extractor(tokens, entity_mention, words) + [0]
                    if len(entity_id)==2:
                        continue
                    '''词性标签'''
                    pos_id = [0] + self.pos_extractor(tokens, single_json, offset) + [0]
                    json_d = {}
                    argument_id=self.argument_and_event_extractor(tokens, event, words)
                    if argument_id==[]:
                        continue
                    argument_id=[0]+argument_id+[0]
                    if padding_length >= 0:
                        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                        attention = [1] * len(tokens) + [0] * padding_length
                        tokens.extend(padding_length * [self.tokenizer.pad_token])
                        entity_id.extend([0] * padding_length)
                        pos_id.extend([0] * padding_length)
                        length = self.max_length - padding_length
                        argument_id.extend([0] * padding_length)

                    if padding_length < 0:
                        tokens = [self.tokenizer.cls_token] + tokens
                        tokens = tokens[:self.max_length]
                        tokens[-1] = self.tokenizer.sep_token
                        attention = [1] * self.max_length
                        entity_id = entity_id[:self.max_length]
                        entity_id[-1] = 0
                        pos_id = pos_id[:self.max_length]
                        pos_id[-1] = 0
                        length = self.max_length
                        argument_id = argument_id[:self.max_length]
                        argument_id[-1] = 0
                    assert len(tokens) == len(entity_id) == len(argument_id) == len(attention) == len(pos_id)
                    tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
                    '''验证一下各个部分是否正确'''
                    json_d['event_type']=event['event_type'].split(":")[1]
                    json_d['length'] = length
                    json_d["tokens"]=tokens
                    json_d['tokens_id']=tokens_id
                    json_d['attention']=attention
                    json_d['entity_id']=entity_id
                    json_d['argument_id']=argument_id
                    json_d['pos_id']=pos_id
                    example.append(json_d)
            else:
                argument_id = [0]*(self.max_length)
                if padding_length >= 0:
                    tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                    attention = [1] * len(tokens) + [0] * padding_length
                    tokens.extend(padding_length * [self.tokenizer.pad_token])
                    entity_id.extend([0] * padding_length)
                    pos_id.extend([0] * padding_length)
                    length = self.max_length - padding_length

                if padding_length < 0:
                    tokens = [self.tokenizer.cls_token] + tokens
                    tokens = tokens[:self.max_length]
                    tokens[-1] = self.tokenizer.sep_token
                    attention = [1] * self.max_length
                    entity_id = entity_id[:self.max_length]
                    entity_id[-1] = 0
                    pos_id = pos_id[:self.max_length]
                    pos_id[-1] = 0
                    length = self.max_length

                assert len(tokens) == len(entity_id) == len(argument_id) == len(attention) == len(pos_id)
                tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
                json_d={}
                json_d['event_type'] = "O"
                json_d['length'] = length
                json_d["tokens"] = tokens
                json_d['tokens_id'] = tokens_id
                json_d['attention'] = attention
                json_d['entity_id'] = entity_id
                json_d['argument_id'] = argument_id
                json_d['pos_id'] = pos_id
                example.append(json_d)
        return example

    def entity_extractor(self,tokens,entity_mentions,words):
        entity_type_start_end_list = []
        if entity_mentions==[]:
            return [0]*len(tokens)
        for mention in entity_mentions:
            text = mention['text']
            text_tokens = self.tokenizer.tokenize(text)  # 重新确定开始结束位置。
            entity_type = mention['entity-type']
            raw_start = mention['start']  # [start,end)左闭右开
            raw_end = mention['end']  # [start,end)左闭右开
            start=0
            end=0
            tmp_token = []
            for i, w in enumerate(words):
                if i == raw_start:
                    start = len(tmp_token)
                if i == raw_end:
                    end = len(tmp_token)
                tmp_token.extend(self.tokenizer.tokenize(w))
            if tmp_token[start:end] != text_tokens:
                # TODO（高天昊）这里有些实例会报玄学错误，不过多数是json数据错误。以后有空修复，目前大部分实例都算正常
                print("raw entity error")
                return []
            entity_type_start_end_list.append((entity_type, start, end))
        entity_id = self.get_tags_id(tokens, entity_type_start_end_list)  # 得到实体的ids,两侧补的0是cls和sep
        return entity_id
    def argument_and_event_extractor(self,tokens,mention,words):
        role_start_end_list = []
        if mention==[]:#没事件，返回全0
            return [0]*len(tokens)
        text = mention['trigger']["text"]
        text_tokens = self.tokenizer.tokenize(text)  # 重新确定开始结束位置。
        raw_start = mention['trigger']['start']  # [start,end)左闭右开
        raw_end = mention['trigger']['end']
        print(words[raw_start:raw_end])
        if text!=' '.join(words[raw_start:raw_end]):
            print("raw data error skip one trigger")
            return []
        tmp_token=[]
        start = 0
        end = 0
        for i, w in enumerate(words):
            if i==raw_start:
                start=len(tmp_token)
            if i==raw_end:
                end=len(tmp_token)
            tmp_token.extend(self.tokenizer.tokenize(w))
        if tmp_token[start:end] != text_tokens:
            # TODO（高天昊）这里有些实例会报玄学错误，不过多数是json数据错误，以后有空修复，目前大部分实例都算正常
            print("skip one argument")
            return []
        role_start_end_list.append(("trigger", start, end))
        argument_id = self.get_tags_id(tmp_token, role_start_end_list, "argument")
        return argument_id
    # def argument_and_event_extractor(self,tokens,mention,offset):
    #     role_start_end_list = []
    #     if mention==[]:
    #         return [0]*len(tokens)
    #     text = mention['trigger']["text"]
    #     text_tokens = self.tokenizer.tokenize(text)  # 重新确定开始结束位置。
    #     raw_start = mention['trigger']['start']  # [start,end)左闭右开
    #     raw_end = mention['trigger']['end']
    #     start=0
    #     end=0
    #     for word_piece_num, word_num in enumerate(offset):
    #         if word_num == raw_start:
    #             start = word_piece_num
    #             continue
    #         if word_num == raw_end:
    #             end = word_piece_num
    #             break
    #         if word_num == raw_end - 1 and word_piece_num == len(offset) - 1:
    #             end = len(offset)
    #             break
    #     if tokens[start:end] != text_tokens:
    #         # TODO（高天昊）这里有些实例会报玄学错误，不过多数是json数据错误，以后有空修复，目前大部分实例都算正常
    #         print("skip one argument")
    #         return None
    #     role_start_end_list.append(("trigger", start, end))
    #     arguments = mention['arguments']
    #     for a in arguments:
    #         text = a['text']
    #         text_tokens = self.tokenizer.tokenize(text)  # 重新确定开始结束位置。
    #         raw_start = a['start']  # [start,end)左闭右开
    #         raw_end=a['end']
    #         start = 0
    #         end = 0
    #         for word_piece_num, word_num in enumerate(offset):
    #             if word_num == raw_start:
    #                 start = word_piece_num
    #                 continue
    #             if word_num == raw_end:
    #                 end = word_piece_num
    #                 break
    #             if word_num == raw_end - 1 and word_piece_num == len(offset) - 1:
    #                 end = len(offset)
    #                 break
    #         if tokens[start:end] != text_tokens:
    #             print("arguments error ,skip one argument")
    #             continue
    #         role_start_end_list.append((a["role"], start, end))
    #     argument_id = self.get_tags_id(tokens, role_start_end_list, "argument")
    #     return argument_id

    def pos_extractor(self,tokens,single_json,offset):
        pos_tags=single_json['pos-tags']
        tokens_pos_tags=np.array([0]*len(tokens))
        word_nums=len(single_json['words'])
        assert (word_nums-1)==offset[-1]
        for i in range(word_nums):
            mask=np.where(np.array(offset)==i)
            tokens_pos_tags[mask]=pos_tag2id[pos2type[pos_tags[i]]]
        return tokens_pos_tags.tolist()

    def get_tags_id(self,tokens,type_start_end_list,type="entity"):
        if type not in ["entity","argument"]:
            raise ValueError("the type should entity or argument")
        if type=='entity':
            tag=[0]*len(tokens)
            for entity in type_start_end_list:
                entity_type,start,end = entity
                entity_type=entity_type.split(":")[0]#粗糙，如果是fine那么就删掉
                for position in range(start,end):
                    tag[position] = entityType2id[entity_type]
            return tag
        if type =="argument":
            BIO = [0] * len(tokens)
            for argumenmts in type_start_end_list:
                role, start, end = argumenmts
                for position in range(start, end):
                    BIO[position] = 1
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

