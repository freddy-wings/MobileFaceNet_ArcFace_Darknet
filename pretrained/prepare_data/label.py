import os
import random

def gen_casia_label(prefix='../../data/CASIA-WebFace-Aligned', 
                    train = '../../data/CASIA_label_train.txt', valid = '../../data/CASIA_label_valid.txt'):
    """
    Notes:
        结果保存在`../../data/CASIA_label_xxxx.txt`，格式为
        `filepath label`
    """
    index = sorted(list(set(os.listdir(prefix))))
    keep = []
    for i in index:
        if len(os.listdir(os.path.join(prefix, i))) > 1:
            keep += [i]
    
    index_dict = dict(zip(keep, range(len(keep))))

    ft = open(train, 'w'); fv = open(valid, 'w')
    for k, v in index_dict.items():
        subdir = os.path.join(prefix, k)
        filenames = os.listdir(subdir)

        valid = random.choice(filenames)
        line = '{:s} {:d}\n'.format('/'.join([k, valid]), v)
        fv.write(line)

        train = filter(lambda x: x not in valid, filenames)
        for filename in train:
            line = '{:s} {:d}\n'.format('/'.join([k, filename]), v)
            ft.write(line)

    ft.close(); fv.close()

if __name__ == "__main__":
    gen_casia_label()
    