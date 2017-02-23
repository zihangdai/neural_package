import torch

def tensorize_binary(padidx, create_mask=False, reverse_seq=False, align_right=False, pin_memory=False):
    def tensorize_binary_core(batch):
        maxlen = max(map(len, batch))
        data = torch.LongTensor(maxlen, len(batch)).fill_(padidx)
        for idx in xrange(len(batch)):
            if reverse_seq:
                record = list(reversed(batch[idx]))
            else:
                record = batch[idx]
            if align_right:
                data[-len(record):, idx] = torch.LongTensor(record)
            else:
                data[:len(record), idx] = torch.LongTensor(record)

        if create_mask:
            mask = torch.ne(data, padidx).float()
            if pin_memory:
                return data.pin_memory(), mask.pin_memory()
            else:
                return data, mask
        else:
            if pin_memory:
                return data.pin_memory()
            else:
                return data
        
    return tensorize_binary_core

def tensorize_string(vocab, create_mask=False, reverse_seq=False, align_right=False, pin_memory=False):
    def tensorize_string_core(batch):
        maxlen = max(map(len, batch))
        data = torch.LongTensor(maxlen, len(batch)).fill_(vocab.pad)
        for idx in xrange(len(batch)):
            if reverse_seq:
                record = map(vocab.encode, reversed(batch[idx]))
            else:
                record = map(vocab.encode, batch[idx])
            if align_right:
                data[-len(record):, idx] = torch.LongTensor(record)
            else:
                data[:len(record), idx] = torch.LongTensor(record)

        if create_mask:
            mask = torch.ne(data, padidx).float()
            if pin_memory:
                return data.pin_memory(), mask.pin_memory()
            else:
                return data, mask
        else:
            if pin_memory:
                return data.pin_memory()
            else:
                return data

    return tensorize_string_core
