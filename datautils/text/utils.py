import numpy
import operator
import os

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

def pack_binary(padidx, create_mask=True, floatX='float32'):
    def pack_binary_core(batch):
        maxlen = max(map(len, batch))
        data = numpy.zeros((maxlen, len(batch)), dtype='int64')
        for idx in xrange(len(batch)):
            record = batch[idx]
            data[:len(record), idx] = record
            data[len(record):, idx] = padidx

        if create_mask:
            mask = numpy.not_equal(data, padidx).astype(floatX)
            return data, mask
        else:
            return data
        
    return pack_binary_core

def pack_string(vocab, create_mask=True, floatX='float32'):
    def pack_string_core(batch):
        maxlen = max(map(len, batch))
        data = numpy.zeros((maxlen, len(batch)), dtype='int64')
        for idx in xrange(len(batch)):
            record = batch[idx]
            data[:len(record), idx] = map(vocab.encode, record)
            data[len(record):, idx] = vocab.pad

        if create_mask:
            mask = numpy.not_equal(data, padidx).astype(floatX)
            return data, mask
        else:
            return data

    return pack_string_core