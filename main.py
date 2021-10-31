import time

doc_split = "\n\n"
sentence_split = "\n"
word_split = " "
text_file = "199801_clear.txt"
stopword = "cn_stopwords.txt"

def read_file(filename):
    file = open(filename, "r", encoding="gbk")
    content = file.read()
    file.close()
    return content

def data_process(content):
    print(content)

def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi("xgy")
