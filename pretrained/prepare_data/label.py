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

def gen_lfw_pairs(oritxt = '../../data/pairs.txt', datapath='../../data/lfw-Aligned', detected = '../../data/lfw_detect.txt',
                    savetxt = '../../data/lfw_pairs.txt'):
    """
    Notes:
    -   根据已有的`pairs.txt`，筛选出检测出人脸的样本
    -   保存在`../../data/lfw_pairs.txt`
    """
    with open(detected, 'r') as f:
        detect_list = f.readlines()
    detect_files = list(map(lambda x: x.split(' ')[0], detect_list))

    with open(oritxt, 'r') as f:
        pairs = f.readlines()
    tosave = [pairs[0]]
    for pair in pairs[1:]:
        p = pair.strip().split('\t')
        if len(p) == 3:
            name, index1, index2 = p
            name1 = name; name2 = name
        else:
            name1, index1, name2, index2 = p
        
        path1 = '{:s}/{:s}/{:s}_{:04d}.jpg'.format(datapath, name1, name1, int(index1))
        path2 = '{:s}/{:s}/{:s}_{:04d}.jpg'.format(datapath, name2, name2, int(index2))

        if (path1 in detect_files) and (path2 in detect_files):
            tosave += [pair]

    with open(savetxt, 'w') as f:
        f.writelines(tosave)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="detect dataset")
    parser.add_argument('--data', '-d', required=True, choices=['casia', 'lfw'])
    args = parser.parse_args()

    if args.data == 'casia':
        gen_casia_label()
    elif args.data == 'lfw':
        gen_lfw_pairs()
    