from __future__ import division
import mmap
import glob
import numpy
import random
import collections

# A thread safe text data reader
class TextReader(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def reset(self, shuffle=False):
        if shuffle:
            self.idx_queue = collections.deque(numpy.random.permutation(len(self)))
        else:
            self.idx_queue = collections.deque(range(len(self)))

    def next_index(self):
        try:
            curr_idx = self.idx_queue.popleft()
        except:
            return None
        return curr_idx

    def next_record(self):
        raise NotImplementedError

class SmallText(TextReader):
    def __init__(self, data_path):
        super(SmallText, self).__init__(data_path)
        
        with open(self.data_path, 'rb') as data_file:
            self.records = data_file.readlines()

        self.reset()

    def next_record(self):
        curr_idx = self.next_index()
        if curr_idx is not None:
            return self.records[curr_idx]
        else:
            return None

    def __len__(self):
        return len(self.records)

class LargeText(TextReader):
    def __init__(self, data_path):
        super(LargeText, self).__init__(data_path)
        
        self.data_file = open(self.data_path, 'r+b')
        self.data_mmap = mmap.mmap(self.data_file.fileno(), 0, access = mmap.ACCESS_READ)

        self.init_mmap()
        self.reset()

    def init_mmap(self):
        self.record_offsets = []
        while True:
            pos = self.data_mmap.tell()
            line = self.data_mmap.readline()
            if line == '':
                break
            self.record_offsets.append(pos)

    def next_record(self):
        curr_idx = self.next_index()
        if curr_idx is not None:
            self.data_mmap.seek(self.record_offsets[curr_idx])
            return self.data_mmap.readline()
        else:
            return None

    def __len__(self):
        return len(self.record_offsets)

    def __del__(self):
        self.data_mmap.close()
        self.data_file.close()

class MultiShardText(object):
    def __init__(self, data_pattern, shuffle_shard=True):
        self.path_list = glob.glob(data_pattern+'*')
        self.shuffle_shard = shuffle_shard

        self.reset()

    def reset(self, shuffle=False):
        self.shuffle = shuffle
        self.shard_idx = -1
        if self.shuffle_shard:
            random.shuffle(self.path_list)
        self.load_next_shard()

    def load_next_shard(self):
        self.shard_idx += 1
        if self.shard_idx >= len(self.path_list):
            self.curr_shard = None
        else:
            self.curr_shard = SmallText(self.path_list[self.shard_idx])
            if self.shuffle: self.curr_shard.reset(self.shuffle)

    def next_record(self):
        # Catch StopIteration and load the next shard
        while self.curr_shard is not None:
            record = self.curr_shard.next_record()
            if record is None:
                self.load_next_shard()
                continue
            return record

        return None

class ParallelText(object):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def next_index(self):
        try:
            curr_idx = self.idx_queue.popleft()
        except:
            return None
        return curr_idx

    def reset(self, shuffle=False):
        if shuffle:
            self.idx_queue = collections.deque(numpy.random.permutation(len(self)))
        else:
            self.idx_queue = collections.deque(range(len(self)))

    def next_record(self):
        raise NotImplementedError

    def __iter__(self):
        return self

class SmallParallelText(ParallelText):
    def __init__(self, data_paths):
        super(SmallParallelText, self).__init__(data_paths)
        
        self.records = []
        for record in zip(*[open(data_path, 'rb') for data_path in self.data_paths]):
            self.records.append(record)

        self.reset()

    def next_record(self):
        curr_idx = self.next_index()
        if curr_idx is not None:
            return self.records[curr_idx]
        else:
            return None

    def __len__(self):
        return len(self.records)

class LargeParallelText(ParallelText):
    def __init__(self, data_paths):
        super(LargeParallelText, self).__init__(data_paths)
        
        self.data_files = [open(data_path, 'r+b') for data_path in self.data_paths]
        self.data_mmaps = [mmap.mmap(data_file.fileno(), 0, access = mmap.ACCESS_READ) for data_file in self.data_files]

        self.init_mmap()
        self.reset()

    def init_mmap(self):
        self.record_offsets = []
        while True:
            pos_tuple = tuple(data_mmap.tell() for data_mmap in self.data_mmaps)
            line_tuple = tuple(data_mmap.readline() for data_mmap in self.data_mmaps)

            if any([line == '' for line in line_tuple]):
                break

            self.record_offsets.append(pos_tuple)

    def next_record(self):
        curr_idx = self.next_index()
        if curr_idx is not None:
            pos_tuple = self.record_offsets[curr_idx]
            record = []
            for data_mmap, pos in zip(self.data_mmaps, pos_tuple):
                data_mmap.seek(pos)
                line = data_mmap.readline()
                record.append(line)

            return tuple(record)
        else:
            return None

    def __len__(self):
        return len(self.record_offsets)

    def __del__(self):
        for data_mmap in self.data_mmaps:
            data_mmap.close()
        for data_file in self.data_files:
            data_file.close()

class MultiShardParallelText(object):
    def __init__(self, data_patterns, shuffle_shard=True):
        self.path_list = zip(*[glob.glob(data_pattern+'*') for data_pattern in data_patterns])
        self.shuffle_shard = shuffle_shard

        self.reset()

    def reset(self, shuffle=False):
        self.shuffle = shuffle
        self.shard_idx = -1
        if self.shuffle_shard:
            random.shuffle(self.path_list)
        self.load_next_shard()

    def load_next_shard(self):
        self.shard_idx += 1
        if self.shard_idx >= len(self.path_list):
            self.curr_shard = None
        else:
            self.curr_shard = SmallParallelText(self.path_list[self.shard_idx], self.shuffle)

    def next_record(self):
        while self.curr_shard is not None:
            record = self.curr_shard.next_record()
            if record is None:
                self.load_next_shard()
                continue
            return record

        return None
