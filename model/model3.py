import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from data_process.constant import event_type, all_slots, event_slot_mask, all_arguments,coarse_grain_entity_BIO_tags
import pandas
import utils


class Event_Model(nn.Module):
    def __init__(self, args):
        super(Event_Model, self).__init__()
        self.args = args
        self.argument_class_num = len(all_arguments)
        self.arguemnt_classifier = nn.Linear(768, 2 * len(all_arguments))
        self.event_type_list = self.args.event_type_list
        # 事件类型embedding
        self.eventType_embedding_path = self.args.filepath + "/event_vec_from_question.pkl"
        self.eventType_embedding = torch.nn.Embedding.from_pretrained(torch.load(self.eventType_embedding_path))
        # self.eventType_embedding = torch.nn.Embedding(len(event_type),768)
        # 实体类型embedding
        self.entity_embedding_path = self.args.filepath + "/entity_vec.pkl"
        self.entity_embedding = torch.nn.Embedding.from_pretrained(torch.load(self.entity_embedding_path))
        # self.entity_embedding = torch.nn.Embedding(len(coarse_grain_entity_BIO_tags),768)
        # slot embedding
        self.slotType_embedding_path = self.args.filepath + "/argument_vec.pkl"
        self.slotType_embedding = torch.nn.Embedding.from_pretrained(torch.load(self.slotType_embedding_path))
        # self.slotType_embedding = torch.nn.Embedding(len(all_slots),768)
        # self.eventType_embedding.weight.requires_grad=True
        self.bert = BertModel.from_pretrained(r'bert-base-uncased')  # D:\斯坦福NLP学习\pretrained
        self.embedding_dim = self.args.embedding_dim
        self.hidden_dim = self.args.hidden_dim
        self.event_bc_list = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 2,bias=True) for _ in range(len(self.event_type_list))])
        self.argument_bc_list = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 2,bias=True) for _ in range(len(all_slots))])

        self.projection = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.argument_projection= nn.Linear( self.embedding_dim, self.hidden_dim,bias=True)
        self.event_projection= nn.Linear( self.embedding_dim, self.hidden_dim,bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.6)
        self.argument_layer_norm=torch.nn.LayerNorm([48, self.hidden_dim])
        self.event_layer_norm=torch.nn.LayerNorm([self.hidden_dim])
        self.loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.attention_weight = torch.nn.Parameter(torch.rand((self.embedding_dim, 1), dtype=torch.double),
                                                   requires_grad=True)

        self.K = nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        self.Q = nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        self.V = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.K1 = nn.Linear(self.embedding_dim, 128)
        self.Q1 = nn.Linear(self.embedding_dim, 128)
        self.V1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.K2 = nn.Linear(self.embedding_dim, 128)
        self.Q2 = nn.Linear(self.embedding_dim, 128)
        self.V2 = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, tokens_id, attention_id, segment_id, entity_id, question, argument_id, event_label=None):
        '''获取bert输出的cls和seq embedding'''
        embedding = self.bert(tokens_id, attention_id, segment_id)
        cls_embedding, seq_embedding = embedding[1], embedding[0]

        event_embedding_index_batch = torch.arange(len(event_type), dtype=torch.long).unsqueeze(0).expand(
            (tokens_id.size(0), len(event_type))).to(self.args.device)
        '''创建实体embedding ，batchsize* maxlen* 768，然后和seq——embedding结合'''
        entity_embedding = self.entity_embedding(entity_id).float()
        if self.args.entity:
            seq_embedding = torch.cat((seq_embedding, entity_embedding), dim=-1)
            # seq_embedding = seq_embedding + entity_embedding.float()
            seq_embedding = self.projection(seq_embedding)

        # event_label = np.array([e_id for e_id in event_label])  # 这里大写，要注意一下
        # event_label_tensor = torch.tensor(event_label, dtype=torch.long, device=self.args.device)
        event_label_tensor = event_label.long()
        '''定义gold label'''
        one_hot_gold_label_1 = torch.nn.functional.one_hot(event_label_tensor, 2)
        '''获取event embedding 并且将cls其扩充为batch * len(event) * 768 维'''
        eventType_embedding = self.eventType_embedding(event_embedding_index_batch).float()
        cls_embedding = cls_embedding.view(cls_embedding.size(0), 1, cls_embedding.size(1)).expand(
            eventType_embedding.size())
        if self.args.dynamic:
            '''每个问题和每个instance拼接，得到batch*len(event）*768的tensor'''
            dynamic_eventtype_embedding = []
            tmp_origin_token = tokens_id[:, 1:]  # 这里把text的id的cls对应的id扔掉,维度为batch*self.max-1
            tmp_origin_attention = attention_id[:, 1:]  # 111100000长度都是相同的
            tmp_origin_segment = segment_id[:, 1:]  # 0000000000长度都是相同的
            question_id, question_attention, question_segment, padding_length, max_question_length = question
            question_element = zip(question_id, question_attention, question_segment, padding_length)
            for q in question_element:
                single_question_id, single_question_attention, single_question_segment, single_question_padding_length = q
                single_question_id = torch.tensor(np.array(single_question_id), dtype=torch.long).unsqueeze(0)
                single_question_id = single_question_id.expand(tmp_origin_token.size(0), single_question_id.size(1)).to(
                    self.args.device)
                single_question_attention = torch.tensor(np.array(single_question_attention),
                                                         dtype=torch.long).unsqueeze(
                    0)
                single_question_attention = single_question_attention.expand(tmp_origin_token.size(0),
                                                                             single_question_attention.size(1)).to(
                    self.args.device)
                single_question_segment = torch.tensor(np.array(single_question_segment), dtype=torch.long).unsqueeze(
                    0)
                single_question_segment = single_question_segment.expand(tmp_origin_token.size(0),
                                                                         single_question_segment.size(1)).to(
                    self.args.device)
                # 一个一个事件拼接处理。
                if single_question_padding_length >= 0:
                    '''先cat一下，然后再最后进行padding'''
                    tmp_concat_id = torch.cat((single_question_id, tmp_origin_token), dim=-1)
                    tmp_concat_attention = torch.cat((single_question_attention, tmp_origin_attention), dim=-1)
                    tmp_concat_segment = torch.cat(
                        (single_question_segment,
                         torch.ones_like(tmp_origin_segment).to(self.args.device)),
                        dim=-1)
                    '''padding应该在最后，也就是在原文的token之后进行padding'''
                    tmp_concat_id = torch.cat((tmp_concat_id, torch.tensor(
                        np.array([self.tokenizer.pad_token_id] * single_question_padding_length)).unsqueeze(0).expand(
                        tmp_concat_id.size(0), single_question_padding_length).to(self.args.device)), dim=-1)
                    tmp_concat_attention = torch.cat((tmp_concat_attention, torch.tensor(
                        np.array([0] * single_question_padding_length)).unsqueeze(0).expand(tmp_concat_id.size(0),
                                                                                            single_question_padding_length).to(
                        self.args.device)), dim=-1)
                    tmp_concat_segment = torch.cat((tmp_concat_segment, torch.tensor(
                        np.array([1] * single_question_padding_length)).unsqueeze(0).expand(tmp_concat_id.size(0),
                                                                                            single_question_padding_length).to(
                        self.args.device)), dim=-1)
                else:
                    '''如果太长，则应该在question部分删除多余的'''
                    tmp_concat_id = torch.cat((single_question_id[:max_question_length], tmp_origin_token), dim=-1)
                    tmp_concat_attention = torch.cat(
                        (single_question_attention[:max_question_length], tmp_origin_attention), dim=-1)
                    tmp_concat_segment = torch.cat((single_question_segment[:max_question_length],
                                                    torch.ones_like(tmp_origin_segment).to(self.args.device)), dim=-1)
                dynamic_seq_embedding, dynamic_cls = self.bert(tmp_concat_id, tmp_concat_attention, tmp_concat_segment)
                dynamic_eventtype_embedding.append(dynamic_cls.cpu().detach().numpy())  # 把一个事件的结果存起来
            dynamic_eventtype_embedding = torch.transpose(torch.tensor(np.array(dynamic_eventtype_embedding)), 1, 0).to(
                self.args.device)
            eventType_embedding = eventType_embedding + 0.5 * dynamic_eventtype_embedding
        # # 2
        # one_hot_gold_label_2 = torch.nn.functional.one_hot(one_hot_gold_label_1, 2)
        '''对slot embedding进行处理'''
        if self.args.slot:
            '''定义slot embedding'''
            slot_embedding_index_batch = torch.arange(len(all_slots), dtype=torch.long).unsqueeze(0).expand(
                (tokens_id.size(0), len(all_slots))).to(self.args.device)
            slotType_embedding = self.slotType_embedding(slot_embedding_index_batch).float()
            '''seq att'''
            slot_key, slot_value = self.K2(slotType_embedding), self.V2(slotType_embedding)
            seq_query = self.Q2(seq_embedding)
            slot_score = torch.bmm(slot_key, torch.transpose(seq_query, 2, 1).float())  # batch * slot type *max len
            scaleed_slot_score = slot_score / (128 ** 0.5)
            score = torch.softmax(scaleed_slot_score, dim=2)
            '''算论元att'''
            seq_key, seq_value = self.K1(seq_embedding), self.V1(seq_embedding)
            slot_query = self.Q1(slotType_embedding)
            seq_score = torch.bmm(seq_key, torch.transpose(slot_query, 2, 1).float())  # batch * max_len, slot_type
            scaleed_seq_score = seq_score / (128 ** 0.5)
            score_t = torch.softmax(scaleed_seq_score, dim=2)
            '''论元'''
            seq_attention_slot_embedding = torch.bmm(score_t, slot_value)
            seq_attention_slot_embedding = seq_attention_slot_embedding + seq_embedding
            label_one_hot = torch.nn.functional.one_hot(argument_id, 2).to(self.args.device)
            argument_logits = self.dropout(seq_attention_slot_embedding)
            loss2=0
            argument_predict=[]
            for i,bc in enumerate(self.argument_bc_list):
                current_argument_logits=self.argument_projection(argument_logits)
                current_argument_logits = self.argument_layer_norm(current_argument_logits)
                current_argument_logits=bc(current_argument_logits)#batch * maxlen *2
                current_argument_logits=self.activation(current_argument_logits)
                loss2+=self.ce_loss(current_argument_logits.view(-1, 2), argument_id[:,:,i].view(-1))
                current_pred=torch.argmax(current_argument_logits, dim=-1).unsqueeze(-1)#batch*maxlength*2-->>batch*maxlen *1
                if argument_predict==[]:
                    argument_predict=current_pred
                else:
                    argument_predict=torch.cat((argument_predict,current_pred),dim=-1)

            '''score矩阵加权，并且加到cls embedding上'''
            slot_attention_seq_embedding = torch.bmm(score, seq_value)  # batch * type * embedding_dim
            batch_slot_attention_seq_embedding = slot_attention_seq_embedding.unsqueeze(1).expand(tokens_id.shape[0],
                                                                                                  len(event_type),
                                                                                                  slot_attention_seq_embedding.shape[
                                                                                                      1],
                                                                                                  slot_attention_seq_embedding.shape[
                                                                                                      2])
            mask = torch.tensor(event_slot_mask).unsqueeze(0).expand(tokens_id.size(0), event_slot_mask.shape[0],
                                                                     event_slot_mask.shape[1]).to(
                self.args.device).float()
            # masked_slot_weight_embedding = torch.bmm(mask,
            #                                          slot_weight_embedding)  # batch * event len* embedding dim
            batch_slot_attention_seq_embedding = mask.unsqueeze(-1).expand_as(batch_slot_attention_seq_embedding) * batch_slot_attention_seq_embedding
            expand_eventType_embedding = eventType_embedding.unsqueeze(2).expand_as(batch_slot_attention_seq_embedding)
            slot_attention_event = torch.sum(batch_slot_attention_seq_embedding * expand_eventType_embedding, dim=-1)
            schema_res = torch.bmm(slot_attention_event, slot_attention_seq_embedding)
            cls_embedding = cls_embedding + schema_res
            # cls_embedding = cls_embedding

            # eventType_embedding = eventType_embedding + masked_slot_weight_embedding
        '''进行对sequence进行attention'''
        if self.args.attention:
            tmp_event_embedding = self.eventType_embedding(event_embedding_index_batch).float()  # 768 * type
            key = self.K(seq_embedding)
            query = self.Q(tmp_event_embedding)
            value = self.V(seq_embedding)
            '''用event embedding与序列中每个token 的embedding计算相似度，得到score矩阵'''
            event2seq_score = torch.bmm(key,
                              torch.transpose(query, 2, 1).float())  # batch * max_len * event_type
            # possible_trigger_idnex=torch.argmax(score,dim=1)
            # possible_trigger_idnex=torch.nn.functional.one_hot(possible_trigger_idnex,seq_embedding.size(1))#应该是16*34*32
            # possible_trigger_embedding=torch.bmm(possible_trigger_idnex.float(),seq_embedding)#batch * event_type * embedding_dim
            scores = event2seq_score.cpu().detach().numpy()
            event2seq_score = torch.softmax(event2seq_score, dim=1)
            event2seq_score = torch.transpose(event2seq_score, 2, 1)
            '''score矩阵加权，并且加到cls embedding上'''
            possible_trigger_embedding = torch.bmm(event2seq_score, seq_embedding)  # batch * event_type * embedding_dim
            cls_embedding = cls_embedding + possible_trigger_embedding
        else:
            scores = 0
        if self.args.static:
            cls_embedding = cls_embedding + 0.5 * eventType_embedding
        # cls_embedding = self.activation(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        total_loss = 0
        alpha = 0.4
        return_logits = []
        logits = []
        '''训练len(event)个二分类器'''
        for i, bc in enumerate(self.event_bc_list):
            current_cls_embedding = cls_embedding[:, i, :]
            current_cls_embedding=self.event_projection(current_cls_embedding)
            current_cls_embedding=self.event_layer_norm(current_cls_embedding)
            current_cls_embedding=self.activation(current_cls_embedding)

            current_label_one_hot = one_hot_gold_label_1[:, i]
            current_logits = bc(current_cls_embedding.float())
            current_logits = self.activation(current_logits)
            # current_logits = self.projection(current_logits)
            '''MSE损失'''
            loss = self.loss(current_logits, current_label_one_hot.float())
            '''交叉熵损失'''
            # loss=self.ce_loss(current_logits.view(-1,2),event_label_tensor[:,i].view(-1))
            '''对于O类，loss权重适当变低'''
            if i == 0:
                total_loss += alpha * loss
            else:
                total_loss += (1 + alpha) * loss
            # return_logits = torch.where(logits > 0.5, torch.scalar_tensor(1).to(self.args.device),
            #                             torch.scalar_tensor(0).to(self.args.device))
            current_return = torch.argmax(current_logits, dim=-1).cpu().detach().numpy()
            return_logits.append(current_return)
            logits.append(current_logits.cpu().detach().numpy())
        return_logits = np.array(return_logits)
        return total_loss,loss2, return_logits.transpose(
            (1, 0)), argument_predict.detach().cpu().numpy(), torch.argmax(one_hot_gold_label_1, dim=-1), torch.argmax(
            label_one_hot, dim=-1), score,score_t,slot_attention_event
