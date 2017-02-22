from random import shuffle
import collections

class BucketIterator(object):
    def __init__(self, dataset, batch_size, cache_size=None, shuffle=False, 
                 process_func=None, sort_func=None, pack_func=None):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.cache_size   = cache_size if cache_size else self.batch_size * 20
        self.process_func = process_func
        self.sort_func    = sort_func
        self.pack_func    = pack_func

    def reset(self):
        self.end_of_epoch = False
        self.dataset.reset(self.shuffle)

    def __iter__(self):
        # reset states
        self.reset()

        while True:
            # caching
            cache = []
            for idx in range(self.cache_size):
                record = self.dataset.next_record()
                if record is None:
                    self.end_of_epoch = True
                    break
                if self.process_func is not None:
                    record = self.process_func(record)
                cache.append(record)

            # sorting
            if self.shuffle and self.sort_func is not None:
                cache = sorted(cache, key = self.sort_func)
            
            # packing
            batches = []
            for idx in range(0, len(cache), self.batch_size):
                batch = cache[idx:idx+self.batch_size]
                if len(batch) == 0:
                    print('Error', idx, len(cache))
                if self.pack_func is not None:
                    batch = self.pack_func(batch)
                batches.append(batch)
            if self.shuffle:
                shuffle(batches)

            # generate batch
            for batch in batches:
                yield batch

            if self.end_of_epoch:
                raise StopIteration
