from tools import CorpusPreprocess, VectorEvaluation
from skip_gram import SkipGram
import torch
from functools import partial
from tqdm import tqdm


class Word2Vec():
    def __init__(self,input_file_path, output_vector_path, neg_num=10, epoches = 10,bach_size = 5,windows_size=5, embedding_dim=300, min_count=5, model_class="skip-gram", optimizer_mothod="huffman"):
        self.min_count = min_count
        self.bach_size = bach_size
        self.neg_num = neg_num
        self.epoches = epoches
        self.windows_size = windows_size
        self.input_file_path = input_file_path
        self.output_vector_path = output_vector_path
        self.data_processor = CorpusPreprocess(self.input_file_path, self.min_count)
        self.input_file_path = input_file_path
        self.embedding_dim = embedding_dim
        self.model_class = model_class
        self.optimizer_mothod = optimizer_mothod
        self.use_cuda = torch.cuda.is_available()
        self.build_model()

    
    def build_model(self):
        if not self.data_processor.vocab:
            self.data_processor.get_vocab()
        if self.model_class == "skip-gram":
            self.model = SkipGram(self.embedding_dim, len(self.data_processor.vocab), self.optimizer_mothod)
        else:
            pass
        if self.use_cuda:
            self.model.cuda()
    
    def train_model(self):
        print("start train !!!")
        optimizer = torch.optim.Adam(self.model.parameters())
        steps = 0
        for epoch in range(self.epoches):
            if self.model_class == "skip-gram":
                data = self.data_processor.build_skip_gram_tain_data(self.windows_size)
            else:
                data = self.data_processor.build_cbow_tain_data(self.windows_size)
            batch_data_iter = self.data_processor.get_bach_data(data, self.bach_size)
            self.data_processor.build_huffman_tree()
            if self.optimizer_mothod == "huffman":
                get_batch_data_fn = self.data_processor.get_bath_huffman_tree_sample
            else:
                get_batch_data_fn = partial(self.data_processor.get_bath_nagative_train_data, count=self.neg_num)
            for batch in batch_data_iter:
                batch_data = get_batch_data_fn(batch)
                pos_v = []
                pos_u = []
                neg_v = []
                neg_u = []
                for i in batch_data:
                    pos_v += i[0]
                    pos_u += i[1]
                    neg_v += i[2]
                    neg_u += i[3]
                input_type = torch.LongTensor
                if self.use_cuda:
                    input_type = torch.cuda.LongTensor
                pos_v = input_type(pos_v)
                pos_u = input_type(pos_u)
                neg_v = input_type(neg_v)
                neg_u = input_type(neg_u)
                loss = self.model(pos_v, pos_u, neg_v, neg_u)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps % 100 == 0:
                    print(f"Epoch {epoch} steps {steps}, loss {loss.item()/len(batch)}")
                    
                steps += 1
        self.save_vector()
        self.evaluation_vector(f"./data/{self.model_class}_{self.optimizer_mothod}.png",
        word="中国", w_num=10)
    
    def save_vector(self):
        if hasattr(self.model, "embedding_matrix"):
            metrix = self.model.embedding_matrix.weight.data
        else:
            metrix = self.model.v_embedding_matrix.weight.data
        with open(self.output_vector_path, "w", encoding="utf-8") as f:
            if self.use_cuda:
                vector= metrix.cpu().numpy()
            else:
                vector = metrix.numpy()
            for i in tqdm(range(len(vector))):
                word = self.data_processor.idex2word[i]
                s_vec = vector[i]
                s_vec = [str(s) for s in s_vec.tolist()]
                write_line = word + " " + " ".join(s_vec)+"\n"
                f.write(write_line)
            print("Word2vec vector save complete!")

    def evaluation_vector(self, picture_path, word, w_num):
        ve = VectorEvaluation(self.output_vector_path)
        ve.drawing_and_save_picture(picture_path)
        ve.get_similar_words(word, w_num)

        
if __name__ == "__main__":
    w2v = Word2Vec(input_file_path="./data/zhihu.txt",output_vector_path="./data/skip-gram.vec", optimizer_mothod="huffman")
    w2v.train_model()