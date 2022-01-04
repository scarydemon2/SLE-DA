import os
import torch
import random
import logging
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
from tqdm import tqdm
from data_process.data_processor_word import ACE05_dataset
from data_process.data_loader import DatasetLoader
from data_process.constant import eventType2id, id2eventType,all_slots
from data_process.constant import event_type as all_event
from model.model3 import Event_Model
import utils
import pandas

logger = logging.getLogger()


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_and_cache_examples(args, processor, data_type='train'):  # 直接从文件中生成torch dataset
    # Load data features from cache or dataset file
    cached_examples_file = os.path.join(args.filepath, 'cached_{}'.format(data_type))
    if os.path.exists(cached_examples_file):
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
        all = 0
        for e in examples:
            type = e['event_type']
            all += np.sum(type[1:])
        print(all)

    else:
        logger.info("Creating features from dataset file at %s", cached_examples_file)
        if data_type == 'train':
            examples = processor.get_train_examples()
        if data_type == 'dev':
            examples = processor.get_dev_examples()
        if data_type == 'test':
            examples = processor.get_test_examples()
        logger.info("Saving features into cached file %s", cached_examples_file)
        torch.save(examples, str(cached_examples_file))
    return examples




def get_question_token():
    import pandas
    question_df = pandas.read_excel(r'data_process/question.xlsx')
    definition = question_df['definition']
    tokenizer = BertTokenizer.from_pretrained(r"bert-base-uncased")
    max = 48
    tokens_id_list, attention_list, segment_list, padding_length_list = [], [], [], []
    for i in range(len(question_df)):
        id = tokenizer.encode_plus(definition[i].replace('\n', ''))['input_ids']
        attention = tokenizer.encode_plus(definition[i].replace('\n', ''))['attention_mask']
        segment = tokenizer.encode_plus(definition[i].replace('\n', ''))['token_type_ids']
        padding_length = max - len(id)
        tokens_id_list.append(id)
        attention_list.append(attention)
        segment_list.append(segment)
        padding_length_list.append(padding_length)
    return tokens_id_list, attention_list, segment_list, padding_length_list, max


def train(args, event_model, processor):
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    # stastics(train_dataset,"train.xlsx")
    train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size, max_length=args.max_length,
                                 shuffle=False, seed=args.seed, sort=False)

    device = args.device
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_loader) * args.epochs

    optimizer = AdamW(event_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # Create the learning rate scheduler.
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
    #                                                           verbose=True, threshold=0.0001, threshold_mode='rel',
    #                                                           cooldown=1, min_lr=0, eps=1e-08)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35, 45], gamma=0.5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    best_result = -1
    for epoch in range(1, args.epochs + 1):
        right = 0.0
        total_right = 0.0
        total = 0.0
        total_p = 0.0
        total_g = 0.0
        argument_right = 0.0
        argument_total_right = 0.0
        argument_total = 0.0
        argument_total_p = 0.0
        argument_total_g = 0.0
        total_loss = 0.0
        notonly_pred = 0
        notonly_gold = 0
        all_zero = 0
        event_model.train()
        question = get_question_token()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            tokens, event_type, tokens_id, attention_mask, segment_id, entity_id, x_len, argument_id = batch
            event_type = event_type.to(device)
            tokens_id = tokens_id.to(device)
            attention_mask = attention_mask.to(device)
            segment_id = segment_id.to(device)
            entity_id = entity_id.to(device)
            argument_id = argument_id.to(device)
            loss, loss2, index, argument_index, gold_onehot, argument_gold_onehot, score, score_t, slot_attention_event = event_model(
                tokens_id,
                attention_mask,
                segment_id,
                entity_id,
                question,
                argument_id,
                event_type)
            # utils.show_attention_heatmap(score.detach().cpu().numpy(), score_t.detach().cpu().numpy(),
            #                              slot_attention_event.detach().cpu().numpy(), event_type.cpu().numpy(),
            #                              gold_onehot.cpu().numpy(), argument_id.cpu().numpy(),
            #                              argument_gold_onehot.cpu().numpy(), tokens)

            # print(loss,loss2)
            index = index.astype(np.long)
            gold_onehot = gold_onehot.cpu().numpy()
            argument_gold_onehot = argument_gold_onehot.cpu().numpy()
            # print(loss,loss2)

            (loss + 2*loss2).backward()
            torch.nn.utils.clip_grad_norm_(event_model.parameters(), 5.0)

            optimizer.step()
            # scheduler.step()
            # lr_scheduler.step()
            for i in range(argument_index.shape[0]):  # 对于一个batch
                for j in range(argument_index.shape[1]):  # 对于batch中的一个token，有36种可能的论元类型。
                    # print(argument_gold_onehot[i][j])
                    # print(argument_index[i][j])
                    argument_total += np.sum(argument_gold_onehot[i][j])
                    argument_total_g += np.sum(argument_gold_onehot[i, j][1:])
                    argument_total_p += np.sum(argument_index[i, j][1:])
                    for k in range(len(argument_gold_onehot[i, j])):
                        if argument_gold_onehot[i, j][k] == argument_index[i, j][k] and argument_index[i, j][k] == 1:
                            argument_total_right += 1
                        if (argument_gold_onehot[i, j, 0] != 1) and argument_gold_onehot[i, j, k] == argument_index[
                            i, j, k] == 1:
                            argument_right += 1

            for i in range(index.shape[0]):
                if np.sum(index[i]) > 1:
                    notonly_pred += 1
                if np.sum(index[i]) == 0:
                    all_zero += 1

                total += np.sum(gold_onehot[i])
                total_g += np.sum(gold_onehot[i][1:])
                total_p += np.sum(index[i][1:])
                for j in range(len(gold_onehot[i])):
                    if gold_onehot[i][j] == index[i][j] == 1:
                        total_right += 1
                    if (gold_onehot[i][0] != 1) and gold_onehot[i][j] == index[i][j] == 1:
                        right += 1
        acc = 1.0 * total_right / (total + 0.000001)
        pre = 1.0 * right / (total_p + 0.000001)
        rec = 1.0 * right / (total_g + 0.000001)
        f1 = 2 * pre * rec / (pre + rec + 0.000001)
        argument_acc = 1.0 * argument_total_right / (argument_total + 0.000001)
        argument_pre = 1.0 * argument_right / (argument_total_p + 0.000001)
        argument_rec = 1.0 * argument_right / (argument_total_g + 0.000001)
        argument_f1 = 2 * argument_pre * argument_rec / (argument_pre + argument_rec + 0.000001)
        print("epoch {} -----------------------------------".format(epoch))
        test_f1 = eval(args, event_model, processor)
        if test_f1 > best_result:
            torch.save(event_model.state_dict(), os.path.join(args.filepath, "argument模型.pkl"))
            best_result = test_f1
        out1 = 'train事件类型：Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (
            total, total_p, total_right, right, total_g)
        out2 = 'train论元：Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total argument: %d\n' % (
            argument_total, argument_total_p, argument_total_right, argument_right, argument_total_g)
        out1 += '事件类型Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (acc, pre, rec, f1)
        out2 += '论元Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (
            argument_acc, argument_pre, argument_rec, argument_f1)
        print(out1)
        print(out2)
        print(all_zero, notonly_pred)

def getSpan(argument_pred,argument_gold):
    import copy
    def span(matrix):
        current_slot_span = []
        for i in range(matrix.shape[0]):  # 对于一个batch
            for k in range(1,len(matrix[i][0])):
                left,right=0,0
                while right < len(matrix[i,:,k]):
                    if matrix[i,left,k]==1:
                        while right<len(matrix[i,:,k]) and matrix[i,right,k]==1:
                            right+=1
                        tmp=[[left,right-1],all_slots[k]]
                        current_slot_span.append(copy.deepcopy(tmp))
                        left=right
                    else:
                        left+=1
                        right=left
        return current_slot_span

    predSpanAndType=span(argument_pred)
    predOnlySpan=[p[0] for p in predSpanAndType]
    goldSpanAndType=span(argument_gold)
    goldOnlySpan = [p[0] for p in goldSpanAndType]
    intersection=0
    type_intersection = 0
    for p in predOnlySpan:
        if p in goldOnlySpan:
            intersection+=1
    for p in predSpanAndType:
        for g in goldSpanAndType:
            if p[0]==g[0] and p[1]==g[1]:
                type_intersection += 1

    return len(predOnlySpan),len(goldOnlySpan),intersection,type_intersection
def microF1(index, label,pred,gold,inter):
    for event in range(1,index.shape[1]):
        for i in range(index.shape[0]):
            if index[i,event]==label[i,event]==1:
                inter[event-1]+=1
        pred[event-1]+=np.sum(index[:,event])
        gold[event-1]+=np.sum(label[:,event])
    return pred, gold, inter

def eval(args, event_model, processor):
    test_dataset = load_and_cache_examples(args, processor, data_type='test')
    # stastics(test_dataset,"test.xlsx")
    test_loader = DatasetLoader(data=test_dataset, batch_size=args.batch_size, max_length=args.max_length,
                                shuffle=False, seed=args.seed, sort=False)
    event_model.eval()
    device = args.device
    optimizer = AdamW(event_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    with torch.no_grad():
        right = 0.0
        total_right = 0.0
        total = 0.0
        total_p = 0.0
        total_g = 0.0
        argument_right = 0.0
        argument_total_right = 0.0
        argument_total = 0.0
        argument_total_p = 0.0
        argument_total_g = 0.0
        identity_right = 0.0
        identity_total_right = 0.0
        identity_total = 0.0
        identity_total_p = 0.0
        identity_total_g = 0.0
        notonly_pred = 0
        micro_pred = [0.00001] * 33
        micro_gold = [0.00001] * 33
        micro_inter = [0] * 33
        notonly_gold = 0
        # e = ErrorAnalysis()
        all_zero = 0
        average_len = 0
        num_notO = 0
        question = get_question_token()
        for i, batch in enumerate(test_loader):
            optimizer.zero_grad()
            tokens, event_type, tokens_id, attention_mask, segment_id, entity_id, x_len, argument_id = batch
            for iii in range(len(x_len.numpy())):
                if event_type[iii][0] != 1:
                    average_len += x_len[iii]
                    num_notO += 1
            event_type = event_type.to(device)
            tokens_id = tokens_id.to(device)
            attention_mask = attention_mask.to(device)
            segment_id = segment_id.to(device)
            entity_id = entity_id.to(device)
            argument_id = argument_id.to(device)
            _, _, index, argument_index, gold_onehot, argument_gold_onehot, score, score_t, slot_attention_event = event_model(
                tokens_id,
                attention_mask,
                segment_id, entity_id,
                question, argument_id,
                event_type)
            # utils.show_attention_heatmap(score.detach().cpu().numpy(), score_t.detach().cpu().numpy(),
            #                              slot_attention_event.detach().cpu().numpy(), event_type.cpu().numpy(),
            #                              gold_onehot.cpu().numpy(), argument_id.cpu().numpy(),
            #                              argument_gold_onehot.cpu().numpy(), tokens)
            index = index.astype(np.long)
            argument_gold_onehot = argument_gold_onehot.cpu().numpy()
            # print(index,gold_onehot)
            gold_onehot = gold_onehot.cpu().numpy().astype(np.long)
            micro_pred,micro_gold,micro_inter=microF1(index,gold_onehot,micro_pred,micro_gold,micro_inter)

            for i in range(index.shape[0]):
                if np.sum(index[i]) > 1:
                    notonly_pred += 1

                if np.sum(index[i]) == 0:
                    all_zero += 1
                # e.update(tokens[i],index[i],gold_onehot[i])
                # if not (index[i] == gold_onehot[i]).all():
                #
                #     print(tokens[i])
                #     print(index[i], gold_onehot[i])
                #     total_different_gold += (gold_onehot[i]!=index[i])+0
                #     total_different_pred += gold_onehot[i]
                total += np.sum(gold_onehot[i])
                total_g += np.sum(gold_onehot[i][1:])
                total_p += np.sum(index[i][1:])
                for j in range(len(gold_onehot[i])):
                    if gold_onehot[i][j] == index[i][j] == 1:
                        total_right += 1
                    if (gold_onehot[i][0] != 1) and gold_onehot[i][j] == index[i][j] == 1:
                        right += 1
                # if (index[i] != gold_onehot[i]).any() and (index[i][0] != 1) and (gold_onehot[i][0] != 1):
                #     print(index[i],gold_onehot[i])
            argument_total_pred,argument_total_gold,intersection,type_intersection=getSpan(argument_index,argument_gold_onehot)
            identity_total_p+=argument_total_pred
            argument_total_p+=argument_total_pred
            identity_total_g+=argument_total_gold
            argument_total_g+=argument_total_gold
            identity_right+=intersection
            argument_right+=type_intersection
            # for i in range(argument_index.shape[0]):  # 对于一个batch
            #     for j in range(argument_index.shape[1]):  # 对于batch中的一个token，有36种可能的论元类型。
            #         argument_total += np.sum(argument_gold_onehot[i, j])
            #         identity_total += np.sum(argument_gold_onehot[i, j])
            #         argument_total_g += np.sum(argument_gold_onehot[i, j][1:])
            #         identity_total_g += 1 if (argument_gold_onehot[i, j][1:]).any() else 0
            #         argument_total_p += np.sum(argument_index[i, j][1:])
            #         identity_total_p += 1 if (argument_index[i, j][1:] == 1).any() else 0
            #         if (argument_gold_onehot[i, j][1:] == 1).any() and (argument_index[i, j][1:] == 1).any():
            #             identity_right += 1
            #         if (argument_gold_onehot[i, j][0] == 1 and argument_index[i, j][0] == 1 and not (
            #                 argument_index[i, j][1:] == 0).any()) \
            #                 or (
            #                 (argument_gold_onehot[i, j][1:] == 1).any() and (argument_index[i, j][1:] == 1).any()):
            #             identity_total_right += 1
            #         for k in range(len(argument_gold_onehot[i, j])):
            #
            #             if argument_gold_onehot[i, j][k] == argument_index[i, j][k] and argument_index[i, j][k] == 1:
            #                 argument_total_right += 1
            #             if (argument_gold_onehot[i, j, 0] != 1) and argument_gold_onehot[i, j, k] == argument_index[
            #                 i, j, k] == 1:
            #                 argument_right += 1
        for i in range(1, 34):
            assert micro_inter[i - 1] <= micro_gold[i - 1]
            p = micro_inter[i - 1] / micro_pred[i - 1]
            r = micro_inter[i - 1] / micro_gold[i - 1]
            f1 = 2 * p * r / (p + r + 1e-5)
            assert p < 1
            assert r < 1
            assert f1 < 1
            print("event_type:{} precision:{},recall:{},f1:{} ".format(all_event[i], p, r, f1))

        # e.write_excel()
        identity_acc = 1.0 * identity_total_right / (identity_total + 0.000001)
        identity_pre = 1.0 * identity_right / (identity_total_p + 0.000001)
        identity_rec = 1.0 * identity_right / (identity_total_g + 0.000001)
        identity_f1 = 2 * identity_pre * identity_rec / (identity_pre + identity_rec + 0.000001)
        argument_acc = 1.0 * argument_total_right / (argument_total + 0.000001)
        argument_pre = 1.0 * argument_right / (argument_total_p + 0.000001)
        argument_rec = 1.0 * argument_right / (argument_total_g + 0.000001)
        argument_f1 = 2 * argument_pre * argument_rec / (argument_pre + argument_rec + 0.000001)

        acc = 1.0 * total_right / (total + 0.000001)
        pre = 1.0 * right / (total_p + 0.000001)
        rec = 1.0 * right / (total_g + 0.000001)
        f1 = 2 * pre * rec / (pre + rec + 0.000001)
        out1 = 'eval事件类型：Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (
            total, total_p, total_right, right, total_g)
        out2 = 'eval论元：Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total argument: %d\n' % (
            argument_total, argument_total_p, argument_total_right, argument_right, argument_total_g)
        out3 = 'eval论元identity：Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total argument: %d\n' % (
            identity_total, identity_total_p, identity_total_right, identity_right, identity_total_g)

        out1 += '事件类型Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (
            acc, pre, rec, f1)
        out2 += '论元Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (
            argument_acc, argument_pre, argument_rec, argument_f1)
        out3 += '论元identity Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (
            identity_acc, identity_pre, identity_rec, identity_f1)
        print(out1)
        print(out2)
        print(out3)
        print(all_zero, notonly_pred)
        return f1 + argument_f1


def evalAndShow(args, event_model, processor):
    test_dataset = load_and_cache_examples(args, processor, data_type='test')
    # stastics(test_dataset,"test.xlsx")
    test_loader = DatasetLoader(data=test_dataset, batch_size=args.batch_size, max_length=args.max_length,
                                shuffle=False, seed=args.seed, sort=False)
    event_model.eval()
    device = args.device
    optimizer = AdamW(event_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    with torch.no_grad():
        right = 0.0
        total_right = 0.0
        total = 0.0
        total_p = 0.0
        total_g = 0.0
        notonly_pred = 0
        notonly_gold = 0
        # e = ErrorAnalysis()
        all_zero = 0
        average_len = 0
        num_notO = 0
        question = get_question_token()
        for i, batch in enumerate(test_loader):
            optimizer.zero_grad()
            tokens, event_type, tokens_id, attention_mask, segment_id, entity_id, x_len, argument_id = batch
            for iii in range(len(x_len.numpy())):
                if event_type[iii][0] != 1:
                    average_len += x_len[iii]
                    num_notO += 1
            event_type = event_type.to(device)
            tokens_id = tokens_id.to(device)
            attention_mask = attention_mask.to(device)
            segment_id = segment_id.to(device)
            entity_id = entity_id.to(device)
            _, index, gold_onehot, scores = event_model(tokens_id, attention_mask, segment_id, entity_id, question,
                                                        event_type)
            # utils.show_attention_heatmap(scores,event_type.cpu().numpy(),gold_onehot.cpu().numpy(),tokens)
            index = index.astype(np.long)
            # print(index,gold_onehot)
            gold_onehot = gold_onehot.cpu().numpy().astype(np.long)
            for i in range(index.shape[0]):
                if gold_onehot[i][0] != 1:
                    '''有事件开始打印'''
                    sentence = tokenizer.convert_tokens_to_string(tokens[i][1:-1])
                    print("当前句子 :", sentence)
                    for k in range(len(gold_onehot[i])):
                        if gold_onehot[i][k] == 1:
                            print("该句子包含事件:", id2eventType[k])
                    for k in range(len(gold_onehot[i])):
                        if index[i][k] == 1:
                            print("探测出来的事件有：", id2eventType[k])
                if np.sum(index[i]) > 1:
                    notonly_pred += 1

                if np.sum(index[i]) == 0:
                    all_zero += 1
                # e.update(tokens[i],index[i],gold_onehot[i])
                # if not (index[i] == gold_onehot[i]).all():
                #
                #     print(tokens[i])
                #     print(index[i], gold_onehot[i])
                #     total_different_gold += (gold_onehot[i]!=index[i])+0
                #     total_different_pred += gold_onehot[i]
                total += np.sum(gold_onehot[i])
                total_g += np.sum(gold_onehot[i][1:])
                total_p += np.sum(index[i][1:])
                for j in range(len(gold_onehot[i])):
                    if gold_onehot[i][j] == index[i][j] == 1:
                        total_right += 1
                    if (gold_onehot[i][0] != 1) and gold_onehot[i][j] == index[i][j] == 1:
                        right += 1
                # if (index[i] != gold_onehot[i]).any() and (index[i][0] != 1) and (gold_onehot[i][0] != 1):
                #     print(index[i],gold_onehot[i])
        # e.write_excel()
        acc = 1.0 * total_right / (total + 0.000001)
        pre = 1.0 * right / (total_p + 0.000001)
        rec = 1.0 * right / (total_g + 0.000001)
        f1 = 2 * pre * rec / (pre + rec + 0.000001)
        out = 'Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (
            total, total_p, total_right, right, total_g)
        out += 'Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (acc, pre, rec, f1)
        print(out)
        print(all_zero, notonly_pred)
        return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default=r"./data", type=str)
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_eval', default=True, action='store_true')
    parser.add_argument("--do_predict", default=True, action='store_true')
    parser.add_argument("--attention", default=True, action='store_true')
    parser.add_argument("--slot", default=True, action='store_true')
    parser.add_argument("--dynamic", default=False, action='store_true')
    parser.add_argument("--static", default=True, action='store_true')
    parser.add_argument("--entity", default=True, action='store_true')
    parser.add_argument('--learning_rate', default=0.000018, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_length', default=48, type=int)
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=48, type=int)
    parser.add_argument("--event_type_list", default=all_event, type=list)
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = "cuda:1"
    else:
        args.device = "cpu"
    seed_everything(1024)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = ACE05_dataset(args.filepath, tokenizer, args.max_length)
    model = Event_Model(args).to(args.device)
    if args.do_train:
        train(args, model, processor)
        torch.save(model.state_dict(), os.path.join(args.filepath, "模型.pkl"))
    if args.do_eval:
        model.load_state_dict(torch.load(os.path.join(args.filepath, r"模型.pkl")))
        eval(args, model, processor)


