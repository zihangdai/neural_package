import os
from .datareader import *

def OneBillionWord(data_dir, load_test=False, load_valid=False):
    data_dir = os.path.expanduser(data_dir)

    trdata_pattern = os.path.join(data_dir,'training-monolingual.tokenized.shuffled/news.en-') 
    tr_set = MultiShardText(data_pattern=trdata_pattern)
    return_value = [tr_set]

    if load_test:
        tedata_pattern = os.path.join(data_dir,'heldout-monolingual.tokenized.shuffled/news.en.heldout-') 
        te_set = MultiShardText(data_pattern=tedata_pattern)
        return_value.append(tr_set)

    if load_valid:
        vadata_path = os.path.join(data_dir,'heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100')
        va_set = SmallText(data_path=vadata_path)
        return_value.append(va_set)
        
    return tuple(return_value)

