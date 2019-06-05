import os
import random

def gen_casia_label(datapath='../../data/CASIA-WebFace-Aligned', 
                    savetxt = '../../data/CASIA_label.txt'):
    """
    Notes:
        结果保存在`../../data/CASIA_label_xxxx.txt`，格式为
        `filepath label`
    """
    index = sorted(list(set(os.listdir(datapath))))
    index_dict = dict(zip(index, range(len(index))))

    f = open(savetxt, 'w')
    for k, v in index_dict.items():
        subdir = os.path.join(datapath, k)
        filenames = os.listdir(subdir)

        for filename in filenames:
            line = '{:s} {:d}\n'.format('/'.join([k, filename]), v)
            f.write(line)

    f.close(); fv.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="detect dataset")
    parser.add_argument('--dir', '-d', default='../../data/CASIA-WebFace-Aligned')
    args = parser.parse_args()

    gen_casia_label(datapath=args.dir)
    