'''分别处理爬取的数据集'''
import os
from config import root_path

def get_classical():
    '''获取经典句子所有数据'''
    contents = []
    data_dir = os.path.join(root_path, 'data/格言网/经典句子/')
    filelist = os.listdir(data_dir)
    print(len(filelist))
    for f in filelist:
        p = data_dir + '/' + f
        with open(p, 'r') as fs:
            lines = fs.readlines()
            for line in lines:
                line = line.strip()
                contents.append(line)
    
    print(len(contents))
    to_file = data_dir + '/' + 'classical.txt'
    with open(to_file, 'w')  as f:
        for con in contents:
            f.write(con + '\n')

def get_encourage():
    '''获取所有名言警句的数据'''
    contents = []
    data_dir = os.path.join(root_path, 'data/格言网/名言警句/')
    filelist = os.listdir(data_dir)
    print(len(filelist))
    for f in filelist:
        p = data_dir + '/' + f
        with open(p, 'r') as fs:
            lines = fs.readlines()
            for line in lines:
                line = line.strip()
                contents.append(line)
    
    print(len(contents))
    to_file = data_dir + '/' + 'encourage.txt'
    with open(to_file, 'w')  as f:
        for con in contents:
            f.write(con + '\n')

def get_love():
    '''获取有关爱情所有的句子'''
    contents = []
    data_dir = os.path.join(root_path, 'data/格言网/爱情格言/')
    filelist = os.listdir(data_dir)
    print(len(filelist))
    for f in filelist:
        p = data_dir + '/' + f
        with open(p, 'r') as fs:
            lines = fs.readlines()
            for line in lines:
                line = line.strip()
                contents.append(line)
    
    print(len(contents))
    to_file = data_dir + '/' + 'love.txt'
    with open(to_file, 'w')  as f:
        for con in contents:
            f.write(con + '\n')






if __name__ == '__main__':
    # get_classical()
    # get_encourage()
    get_love()
