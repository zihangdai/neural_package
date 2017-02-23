import numpy

def tuplize_share(func, batch=False):
    def tuplize_share_core(record):
        if batch:
            record = zip(*record)
        output = tuple(func(field) for field in record)
        return output
    return tuplize_share_core

def tuplize_sep(funcs, batch=False):
    def tuplize_sep_core(record):
        if batch:
            record = zip(*record)
        if len(funcs) != len(record):
            raise ValueError('[len(funcs) = {}] != [len(record) = {}]'.format(len(funcs), len(record)))
        output = tuple(func(field) for func, field in zip(funcs, record))
        return output
    return tuplize_sep_core

def split_line(line):
    return line.strip().split()

def binarize(vocab, func):
    def binarize_core(record):
        symbols = func(record)
        indices = [vocab.encode(sym) for sym in symbols]
        return indices

    return binarize_core

def pack_binary(padidx, create_mask=False, reverse_seq=False, align_right=False, floatX='float32'):
    def pack_binary_core(batch):
        maxlen = max(map(len, batch))
        data = numpy.zeros((maxlen, len(batch)), dtype='int64')
        data.fill(padidx)
        for idx in xrange(len(batch)):
            if reverse_seq:
                record = list(reversed(batch[idx]))
            else:
                record = batch[idx]
            if align_right:
                data[-len(record):, idx] = record
            else:
                data[:len(record), idx] = record

        if create_mask:
            mask = numpy.not_equal(data, padidx).astype(floatX)
            return data, mask
        else:
            return data
        
    return pack_binary_core

def pack_string(vocab, create_mask=False, reverse_seq=False, align_right=False, floatX='float32'):
    def pack_string_core(batch):
        maxlen = max(map(len, batch))
        data = numpy.zeros((maxlen, len(batch)), dtype='int64')
        data.fill(vocab.pad)
        for idx in xrange(len(batch)):
            if reverse_seq:
                record = map(vocab.encode, reversed(batch[idx]))
            else:
                record = map(vocab.encode, batch[idx])
            if align_right:
                data[-len(record):, idx] = record
            else:
                data[:len(record), idx] = record

        if create_mask:
            mask = numpy.not_equal(data, padidx).astype(floatX)
            return data, mask
        else:
            return data

    return pack_string_core
