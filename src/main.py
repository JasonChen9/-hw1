from os import read
import time
import re
import math 
from multiprocessing import Pool, cpu_count, Manager
import numpy as np
from gensim.models.word2vec import KeyedVectors
from collections import Counter

#Global variable
##Delimiter 
doc_split = "\n\n"
sentence_split = "\n"
word_split = " "
##file path
text_file = "../data/199801_clear.txt"
stop_word = "../data/cn_stopwords.txt"
embedding_table = "../../Tencent_AILab_ChineseEmbedding.txt"
embedding_table_bin = "../../ChineseEmbedding.bin"
def ReadFile(filename):
    """Read text file 

    Args:
        filename ([string]): [the path of the text file]

    Returns:
         lines ([list]): [Return all lines in the file, as a list where each line is an item in the list object]
    """
    file = open(filename, "r", encoding="gbk")
    lines = file.readlines()
    file.close()
    return lines

def ReadStopword(stopword_file):
    """Read stopword table 

    Args:
        stopword_file ([string]): [the path of the stop word file]

    Returns:
        stop_words ([list]): [Return all words in the file, as a list where each word is an item in the list object]
    """
    file = open(stopword_file)
    stop_words = file.read()
    stop_words = stop_words.split('\n')  
    file.close()
    return stop_words


def DataProcess(lines, stop_words):
    """Fill raw data in a docs list 
    and each item in the list is a doc, as a list where each word is an item in the list object

    Args:
        lines ([list]): [all lines in the text file, as a list where each line is an item in the list object]
        stop_words ([list]): [all words in the stopword file, as a list where each word is an item in the list object]

    Returns:
        docs ([list]): [doc in list]
    """
    re_pattner = re.compile(u"[][\u4e00-\u9fa5]+")    
    pre_id = lines[0][:15]
    doc = []
    docs = []

    for line in lines:
        if line == '\n':
            continue
        now_id=line[0:15]
        word_list = re.findall(re_pattner, line)
        for word in word_list:
            if word in stop_words:
                word_list.remove(word)

        if now_id == pre_id:
            doc += word_list
        else:
            countlist = Counter(doc)
            docs.append(countlist)
            doc = word_list
            pre_id = now_id
               
    countlist = Counter(doc)    
    docs.append(countlist)
    return docs

def tf(word, doc):
    return doc[word] / sum(doc.values())
        
def df(word, docs):
    cnt = 0
    for doc in docs:
        if word in doc:
            cnt+=1
    
    return cnt

def idf(word, docs):
    return math.log(len(docs)/df(word,docs)+1, 10)

def get_tf_idf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)


def tf_idf_proc(index, doc, dict, docs):
    scores = {word: get_tf_idf(word, doc, docs) for word in doc}    
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    dict[index] = sorted_words[0:10]

def TfIdf(docs):
    tf_idf_start = time.time()
    tf_idf_list = []
    pool = Pool(cpu_count())
    tf_idf_dict = Manager().dict()
    for index, doc in enumerate(docs):
        pool.apply_async(tf_idf_proc, args=(index, doc, tf_idf_dict, docs))

    pool.close()
    pool.join()    
    tf_idf_end = time.time()
    print("calculate ifidf finished, spent :", tf_idf_end - tf_idf_start, "s")
    for i in range(len(tf_idf_dict)):
        tf_idf_list.append(dict(tf_idf_dict[i]))
    return tf_idf_list

def save_embedding(embedding_file):
    save_start = time.time()
    wv_from_text = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    print("start save")
    wv_from_text.save("../../ChineseEmbedding.bin")
    save_end = time.time()
    print("embedding file save finished, spent :", save_end-save_start, "s")


def load_embedding(embedding_file):
    load_start = time.time()
    wv_from_bin = KeyedVectors.load(embedding_file, mmap='r')
    load_end = time.time()
    print("embedding file load finished, spent :", load_end-load_start, "s")
    return wv_from_bin

def doc2vec(doc,wv_from_bin):
    doc_vec=[]
    for word in doc:
         if word in wv_from_bin.index_to_key:
             doc_vec.append(wv_from_bin[word])

    if len(doc_vec)!= 0:
        return np.sum(doc_vec, axis=0) / len(doc_vec)
    else:
        return np.zero(200)

def cosine_similarity(vector1, vector2):
    assert (len(vector1) == len(vector2))

    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]
        normA += vector1[i] ** 2
        normB += vector2[i] ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA ** 0.5) * (normB ** 0.5)), 5)

def cal_similarity_matrix(i, dict,tf_idf_list,doc_vec):
    similarity_array = np.zeros(len(tf_idf_list))
    for j in range(len(tf_idf_list)):
        if i == j:
            similarity_array[j] = 1
        elif i > j:
            similarity_array[j] = cosine_similarity(doc_vec[i],doc_vec[j])
    dict[i] = similarity_array


def Similarity(tf_idf_list, doc_vec):
    similarity_start = time.time()
    similarity_matrix = []
    np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
    
    pool = Pool(cpu_count())
    similarity_dict = Manager().dict()

    for i in range(len(tf_idf_list)):
         pool.apply_async(cal_similarity_matrix, args=(i, similarity_dict,tf_idf_list,doc_vec))

    pool.close()
    pool.join()
    similarity_dict = dict(similarity_dict)
    similarity_end = time.time()
    print("The time of calculating similarity matrix:",similarity_end - similarity_start)
    for i in range(len(tf_idf_list)):
        similarity_matrix.append(similarity_dict[i])
    return similarity_matrix

if __name__ == '__main__':
    
    lines = ReadFile(text_file)
    stop_words = ReadStopword(stop_word)
    docs = DataProcess(lines, stop_words)
    docs_test = docs[0:1000]

    #TfIdf
    tf_idf_list = TfIdf(docs_test)
    wv_from_bin = load_embedding(embedding_table_bin)
    
    doc_vec = []
    for doc in tf_idf_list:
        doc_vec.append(doc2vec(doc.keys(), wv_from_bin))
    
    Similarity(tf_idf_list,doc_vec)
   






