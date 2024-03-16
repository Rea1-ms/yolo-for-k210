import os
import re
import numpy as np
import sys
import argparse
import skimage


def main(train_file: str, output_file: str):
    image_path_list = np.loadtxt(train_file, dtype=str)

    if not os.path.exists('data'):
        os.makedirs('data')

    ann_list = list(image_path_list)
    ann_list = [re.sub(r'JPEGImages', 'labels', s) for s in ann_list]
    ann_list = [re.sub(r'.jpg', '.txt', s) for s in ann_list]

    # 使用列表而不是NumPy数组来存储信息
    lines = [
        [
            image_path_list[i],
            np.loadtxt(ann_list[i], dtype=float, ndmin=2) if os.path.exists(ann_list[i]) else np.empty((0, 5)),
            np.array(skimage.io.imread(image_path_list[i]).shape[0:2])
        ] for i in range(len(ann_list))
    ]

    # 保存数据
    # 注意：由于lines现在是一个列表，不是所有元素都是NumPy数组，因此不能直接使用np.save
    # 可以考虑使用pickle来保存复杂的数据结构
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(lines, f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='trian.txt file path')
    parser.add_argument('output_file', type=str, help='output file path')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args.train_file, args.output_file)
