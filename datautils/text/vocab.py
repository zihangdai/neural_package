import collections

class Vocab(object):
    def __init__(self, counter, max_size=None, min_freq=1,
                 special_syms = ['<unk>', '<pad>']):
        self.sym2idx = collections.OrderedDict()
        self.idx2sym = []

        for sym in special_syms:
            self.sym2idx[sym] = len(self.idx2sym)
            self.idx2sym.append(sym)

        for sym, freq in counter.most_common(max_size):
            if freq >= min_freq:
                self.sym2idx[sym] = len(self.idx2sym)
                self.idx2sym.append(sym)

    @property
    def unk(self):
        return self.sym2idx['<unk>']

    @property
    def pad(self):
        return self.sym2idx['<pad>']

    def encode(self, sym):
        return self.sym2idx.get(sym, self.unk)

    def decode(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError('index {} is out of range'.format(idx))
        return self.idx2sym[idx]

    def __len__(self):
        return len(self.idx2sym)