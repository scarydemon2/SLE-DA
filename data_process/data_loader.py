import torch
import numpy as np
import random
from constant import eventType2id
class DatasetLoader(object):
    def __init__(self, data, batch_size, max_length,shuffle,seed, sort=True):
        self.data = data#[{}、{}、{}]
        self.shuffle = shuffle
        self.max_length=max_length
        self.batch_size = batch_size
        self.seed = seed
        self.sort = sort
        self.reset()

    def reset(self):
        self.examples = self.preprocess(self.data)
        if self.sort:
            self.examples = sorted(self.examples, key=lambda x: x[4], reverse=True)
        if self.shuffle:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        self.features = [self.examples[i:i + self.batch_size] for i in range(0, len(self.examples), self.batch_size)]
        print(f"{len(self.features)} batches created")

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['tokens']
            tokens_id=d["tokens_id"]
            attention_mask=d['attention']
            entity_id=d['entity_id']
            argument_id=d['argument_id']
            x_len = d['length']#未padding之前的长度
            event_type_id = d['event_type']
            assert len(tokens_id)==len(attention_mask)==self.max_length
            processed.append((tokens, tokens_id, attention_mask,entity_id,x_len, event_type_id,argument_id))
        return processed

    # def get_long_tensor(self, tokens_list, batch_size, mask=None):
    #     """ Convert list of list of tokens to a padded LongTensor. """
    #     token_len = max(len(x) for x in tokens_list)
    #     tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    #     mask_ = torch.LongTensor(batch_size, token_len).fill_(0)
    #     for i, s in enumerate(tokens_list):
    #         tokens[i, :len(s)] = torch.LongTensor(s)
    #         if mask:
    #             mask_[i, :len(s)] = torch.tensor([1] * len(s), dtype=torch.long)
    #     if mask:
    #         return tokens, mask_
    #     return tokens

    def sort_all(self, batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """
        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    def __len__(self):
        # return 50
        return len(self.features)

    def __getitem__(self, index):
        """ Get a batch with index. """
        if not isinstance(index, int):
            raise TypeError
        if index < 0 or index >= len(self.features):
            raise IndexError
        batch = self.features[index]
        batch_size = len(batch)
        batch = list(zip(*batch))
        lens = [x for x in batch[4]]
        batch, orig_idx = self.sort_all(batch, lens)
        tokens = batch[0]
        tokens_id=torch.tensor(np.array(batch[1])).long()
        attention_mask=torch.tensor(np.array(batch[2])).long()
        entity_id=torch.tensor(np.array(batch[3])).long()
        x_len=torch.tensor(np.array(batch[4])).long()
        argument_id=torch.tensor(np.array(batch[6])).long()
        segment_id = torch.zeros(size=(attention_mask.shape)).long()
        event_type_id=torch.tensor(np.array(batch[5]))
        return (tokens,event_type_id,tokens_id,attention_mask,segment_id,entity_id,x_len,argument_id)