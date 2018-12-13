import os
import logging
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from scipy.sparse import lil_matrix
from collections import Counter

encoding_type = "utf-8"
split_sep = " "


class CorpusPreprocess(object):
    logger = logging.getLogger("CorpusPreprocess")

    def __init__(self, file_path, min_count):
        self.file_path = file_path
        self.min_count = min_count
        self.vocab = Counter()
        self.cooccurrence_matrix = None
        self.idex2word = None
        self.word2idex = None
    
    def _read_data(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"file path {self.file_path} is not exist !")
        with open(self.file_path, "r", encoding = encoding_type) as f:
            for line in f:
                if line.strip():
                    yield line.strip().split(split_sep)
    
    def _build_vocab(self):
        for line in tqdm(self._read_data()):
            self.vocab.update(line)
        self.vocab = dict((w.strip(), f) for w,f in self.vocab.items() if (f >= self.min_count and w.strip()))
        self.vocab = {w:(i, f) for i, (w, f) in enumerate(self.vocab.items())}
        self.idex2word = {i:w for w, (i,f) in self.vocab.items()}
        self.logger.info("build vocab complete!")

    def _build_cooccurrence_matrix(self, windows_size=5):
        if not self.vocab:
            self._build_vocab()
        self.cooccurrence_matrix = lil_matrix((len(self.vocab), len(self.vocab)),dtype=np.float32)
        for line in tqdm(self._read_data()):
            sentence_length = len(line)
            for i in range(sentence_length):
                center_w = line[i]
                if center_w not in self.vocab:
                    continue
                left_ws = line[max(i-windows_size,0):i]
                # right_ws = line[i+1:min(len(line),i+1+windows_size)]
                
                # left cooccur
                for i, w in enumerate(left_ws[::-1]):
                    if w not in self.vocab:
                        continue
                    self.cooccurrence_matrix[self.vocab[center_w][0],
                                             self.vocab[w][0]] += 1.0 / (i+1.0)
                    # cooccurrence_matrix is Symmetric Matrices
                    self.cooccurrence_matrix[self.vocab[w][0],
                                             self.vocab[center_w][0]] += 1.0 / (i+1.0)
                # for i, w in enumerate(right_ws):
                    
                #     self.cooccurrence_matrix[self.vocab[center_w][0],
                #                              self.vocab[w][0]] += 1.0 /(i+1.0)

                        
        self.logger.info("build cooccurrece matrix complete!")
    
    def get_cooccurrence_matrix(self, windows_size):
        if self.cooccurrence_matrix == None:
            self._build_cooccurrence_matrix(windows_size)
        return self.cooccurrence_matrix
    
    def get_vocab(self):
        if self.vocab == None:
            self._build_vocab()
        return self.vocab
        


class VectorEvaluation(object):
    def __init__(self, vector_file_path):
        if os.path.exists(vector_file_path):
            self.vector_file_path = vector_file_path
        else:
            raise FileExistsError("file is not exists!")
        self.read_data()
    
    def _read_line(self, word, *vector):
        return word, np.asarray(vector, dtype=np.float32)

    def read_data(self):
        words = []
        vector = []
        with open(self.vector_file_path, "r", encoding=encoding_type) as f:
            for line in f:
                word, vec = self._read_line(*line.split(split_sep))
                words.append(word)
                vector.append(vec)
        assert len(vector) == len(words)
        self.vector = np.vstack(tuple(vector))
        self.vocab = {w:i for i,w in enumerate(words)}
        self.idex2word = {i:w for w,i in self.vocab.items()} 

    def drawing_and_save_picture(self, picture_path, w_num=10, mode="pca"):
        w_num = min(len(self.vocab), w_num)
        reduce_dim = PCA(n_components=2)
        if mode == "tsne":
            reduce_dim = TSNE(n_components=2)
        vector = reduce_dim.fit_transform(self.vector)
        idex = np.random.choice(np.arange(len(self.vocab)), size=w_num, replace=False)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        for i in idex:
            plt.scatter(vector[i][0], vector[i][1])
            plt.annotate(self.idex2word[i], xy=(vector[i][0], vector[i][1]),xytext=(5, 2),textcoords='offset points')
        plt.title(f"{mode} - " + picture_path.split("/")[-1].split(".")[0])
        plt.savefig(picture_path)
        print(f"save picture to {picture_path}")

    def get_similar_words(self, word, w_num=10):
        w_num = min(len(self.vocab), w_num)
        idx = self.vocab.get(word,None)
        if not idx:
            return
        result = cosine_similarity(self.vector[idx].reshape(1,-1), self.vector)
        result = np.array(result).reshape(len(self.vocab),)
        idxs = np.argsort(result)[::-1][:w_num]
        print("<<<"*7)
        print(word)
        for i in idxs:
            print("%s : %.3f\n" % (self.idex2word[i], result[i]))
            
        print(">>>" * 7)


        
