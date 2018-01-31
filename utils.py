# -*- coding: utf-8 -*-

class CharEmbeddedEncoder:
    """
    An encoder for character embedding based on "Text Understanding from Scratch"
        URL: https://arxiv.org/pdf/1502.01710.pdf
    """
    np = __import__('numpy')
    mp = __import__('multiprocessing')
    def __init__(self, n_jobs=2, sequence_max_length=1014):
        self.alphabet =  'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/|_#$%^&*~`+=<>()[]{}\\ \n'
        self.char_dict = {}
        self.sequence_max_length = sequence_max_length
        self.n_jobs = n_jobs
        for i,c in enumerate(self.alphabet):
            self.char_dict[c] = i
        self.char_dict_len = len(self.char_dict)+1
                          
    def char2vec(self, text):
        data = self.np.ones(self.sequence_max_length) * self.char_dict_len
        for i in range(len(text)):
            if text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                data[i] = self.char_dict_len - 1
            if i > self.sequence_max_length:
                return data
        return data
    
    def transform(self, documents):
        char_vecs = []
        for document in documents:
            char_vecs.append(self.char2vec(document))
        return self.np.asarray(char_vecs, dtype=int)