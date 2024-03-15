import os
import json
import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--device', type=str, default=r'cuda:0', help='which gpu to use')

parser.add_argument('--columns', type=str, default=r'contet', help='input columns name')
# read xlsx
parser.add_argument('--model', type=str, default=r'resultModel', help='model path')
parser.add_argument('--xlsx_directory', type=str, default=r'./excel', help='input excel directory')
# merge file
parser.add_argument('--directory', type=str, default=r'./csv', help='output excel directory')
# parser.add_argument('--file_name', type=str, default=r'all.csv', help='save file name')
# pall_func
parser.add_argument('--threshold', type=float, default=0.85, help='threshold value')

args = parser.parse_args()


def vec(sentence_list):
    sentence_list = np.array(sentence_list[args.columns].values.tolist())
    sentence_list = sentence_list[sentence_list != 'nan']
    # 加载模型

    embedding_list = model.encode(sentence_list)
    # 存储

    vector = []
    for i in range(len(sentence_list)):
        # entity_vectors[sentence_list[i]] = embedding_list[i].tolist()
        vector.append(embedding_list[i].tolist())
    return vector


def read():
    directory = args.xlsx_directory
    files = os.listdir(directory)
    da = pd.DataFrame([])

    for file in files:
        file_name = '/' + file.split(sep='.')[0]
        da1 = pd.read_excel(io=directory + file_name + '.xlsx').dropna(subset=[args.columns], inplace=False)
        da1['体系'] = da1['体系'][0]
        da1 = da1.replace('\xa0', '', regex=True)
        da = da.append(da1)
    da = da[['体系', '标题', '标准名称', '标准编号', args.columns]]
    grouped = da.groupby('体系')

    for name, group in grouped:
        filename = f"csv\\{name}.csv"
        group.to_csv(filename, index=False)


def merge_file():
    directory = args.directory
    files = os.listdir(directory)
    for file in files:
        data = pd.read_csv(directory + '\\' + file)
        vector = vec(data)
        res = pall_func(np.array(vector))

        # 生成标准编号的唯一值列表，保持原始顺序
        unique_standard_numbers = pd.Series(data['标准编号'])

        # 创建一个全为0的矩阵，行数和列数都是唯一标准编号的数量
        matrix_size = len(unique_standard_numbers)
        result_matrix = np.zeros((matrix_size, matrix_size))

        # 遍历DataFrame，将相同标准编号的位置设置为1
        for index, row in data.iterrows():
            i = np.where(unique_standard_numbers != row['标准编号'])[0]
            result_matrix[index, i] = 1
        res = res * result_matrix
        data1 = {str(i): [';'.join(map(str, np.where(row == 1)[0]))] for i, row in
                 enumerate(res - np.eye(res.shape[0], res.shape[1]))}
        df = pd.DataFrame.from_dict(data1).T
        data['相似条款索引'] = df[0].to_list()
        data.to_csv(directory + '\\' + file, index=False)


def pall_func(vectors):
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine_similarity = np.dot(vectors, vectors.T) / (vector_norms * vector_norms.T)

    threshold = args.threshold
    cosine_similarity[cosine_similarity >= threshold] = 1
    cosine_similarity[cosine_similarity < threshold] = 0
    return cosine_similarity


if __name__ == '__main__':
    model = SentenceTransformer(args.model).to(args.device)
    read()
    merge_file()

