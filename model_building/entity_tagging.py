# -*- coding: utf-8 -*-
import re
import codecs

date_pattern = []
phone_pattern = []
link_pattern = []
currency_pattern = []
emoticon_pattern = []

date_string = ' date '
phone_string = ' phone '
link_string = ' link '
currency_string = ' currency '
emoticon_string = ' emoticon '
# Regex for date
date_pattern.append(r'\d{1,2}/\d{1,2}/\d{2,4}')
date_pattern.append(r'\d{1,2}-\d{1,2}-\d{2,4}')
date_pattern.append(r'\+\d{1,2}:\d{1,4}')
date_pattern.append(r'\d{1,2}/\d{1,4}')
date_pattern.append(r'\d{1,2}-\d{1,4}')
date_pattern.append(r'\d{1,2}h\d{0,2}')
# Regex for phone
phone_pattern.append(r'\+\d{10,12}')
phone_pattern.append(r'\d{3,5}\.\d{3,4}\.\d{3,5}')
phone_pattern.append(r'\d{8,12}')
phone_pattern.append(r'1800\d{4}')
phone_pattern.append(r'195')
phone_pattern.append(r'900')
phone_pattern.append(r'1342')
phone_pattern.append(r'191')
phone_pattern.append(r'888')
phone_pattern.append(r'333')
phone_pattern.append(r'1414')
phone_pattern.append(r'1576')
phone_pattern.append(r'8170')
phone_pattern.append(r'9123')
phone_pattern.append(r'9118')
phone_pattern.append(r'266')
phone_pattern.append(r'153')
phone_pattern.append(r'199')
phone_pattern.append(r'9029')
phone_pattern.append(r'8049')
phone_pattern.append(r'1560')
phone_pattern.append(r'9191')
# Regex for link
link_pattern.append(r'www\..*')
link_pattern.append(r'http://.*')
# Regex for currency
currency_pattern.append(r'[0-9|\,\.]{3,}VND')
currency_pattern.append(r'[0-9|\.]{3,}VND')
currency_pattern.append(r'[0-9|\.]{3,}d')
currency_pattern.append(r'[0-9|\.]{3,}đ')
currency_pattern.append(r'[0-9|\.]{3,}tr')
currency_pattern.append(r'[0-9|\.]{3,}Tr')
currency_pattern.append(r'[0-9|\.]{3,}TR')
# Regex for emoticon
emoticon_pattern.append(r'o.O')
emoticon_pattern.append(r'O.o')
emoticon_pattern.append(r'\(y\)')
emoticon_pattern.append(r'\(Y\)')
emoticon_pattern.append(r':v')
emoticon_pattern.append(r':V')
emoticon_pattern.append(r':3')
emoticon_pattern.append(r'-_-')
emoticon_pattern.append(r'\^_\^')
emoticon_pattern.append(r'<3')
emoticon_pattern.append(r':-\*')
emoticon_pattern.append(r':\*')
emoticon_pattern.append(r":'\(")
emoticon_pattern.append(r':p ')
emoticon_pattern.append(r':P')
emoticon_pattern.append(r':d')
emoticon_pattern.append(r':D')
emoticon_pattern.append(r':-\?')
emoticon_pattern.append(r'>\.<')
emoticon_pattern.append(r'><')
emoticon_pattern.append(r':-\w ')
emoticon_pattern.append(r':\)\)')
emoticon_pattern.append(r';\)\)')
emoticon_pattern.append(r'=\)\)')
emoticon_pattern.append(r':-\)')
emoticon_pattern.append(r':\)')
emoticon_pattern.append(r':\]')
emoticon_pattern.append(r'=\)')
emoticon_pattern.append(r':-\(')
emoticon_pattern.append(r':\(')
emoticon_pattern.append(r':\[')
emoticon_pattern.append(r'=\(')

stop_list = ['.', ',', '/', '?', ';', ':', '&', '@', '!', '`', "'", '"', '>', '<', '*', '%', '#', '(', ')', '[', ']',
             '-', '_', '=', '+', '{', '}', '~', '$', '^', '*', '|', '\\']

# reader = codecs.open('corpus.txt', 'r', 'utf8')
# writer = codecs.open('res.txt', 'w', 'utf8')
#
# for line in reader.readlines():
#     line = line.strip()
#     temp = line.split('\t')
#     label = temp[0]
#     line = temp[1]
#     sentence = ''
#     for word in line.split(' '):
#         for date_pat in date_pattern:
#             word = re.sub(date_pat, date_string, word)
#         for currency_pat in currency_pattern:
#             word = re.sub(currency_pat, currency_string, word)
#         for phone_pat in phone_pattern:
#             word = re.sub(phone_pat, phone_string, word)
#         for link_pat in link_pattern:
#             word = re.sub(link_pat, link_string, word)
#         for emoticon_pat in emoticon_pattern:
#             word = re.sub(emoticon_pat, emoticon_string, word)
#         sentence += word + ' '
#     writer.write(label + '\t' + sentence.strip() + '\n')
# reader.close()
# writer.close()
#
# f1 = codecs.open('res.txt', 'r', 'utf-8')
# f2 = codecs.open('corpus_train.txt', 'w', 'utf-8')
# list_label = []
# list_content = []
# for line in f1:
#     line = line.split('\t')
#     list_label.append(line[0])
#     temp = line[1]
#     stop_list = ['.', ',', '/', '?', ';', ':', '&', '@', '!', '`', "'", '"', '>', '<', '*', '%', '#', '(', ')', '[',
#                  ']', '-', '_', '=', '+', '{', '}', '~', '$', '^', '*', '|', '\\']
#     for item in stop_list:
#         temp = temp.replace(item, ' ')
#     list_content.append(temp.strip('\n').lower())
# count = 0
# for line in list_content:
#     sentence = ''
#     line = line.split()
#     for word in line:
#         if word.isdigit():
#             word = ' number '
#         sentence += word + ' '
#     f2.write(list_label[count] + '\t' + sentence.strip() + '\n')
#     count += 1
# f1.close()
# f2.close()


def entity_tagging(corpus):
    corpus_new = []
    for line in corpus:
        sent = []
        for word in line.split():
            for date_pat in date_pattern:
                word = re.sub(date_pat, date_string, word)
            for currency_pat in currency_pattern:
                word = re.sub(currency_pat, currency_string, word)
            for phone_pat in phone_pattern:
                word = re.sub(phone_pat, phone_string, word)
            for link_pat in link_pattern:
                word = re.sub(link_pat, link_string, word)
            for emoticon_pat in emoticon_pattern:
                word = re.sub(emoticon_pat, emoticon_string, word)
            sent.append(word)
        sent = ' '.join(sent)
        for item in stop_list:
            sent = sent.replace(item, ' ')
        sent = sent.lower()
        sent = sent.split()
        for i in range(len(sent)):
            if sent[i].isdigit():
                sent[i] = 'number'
        corpus_new.append(' '.join(sent))
    return corpus_new


if __name__ == '__main__':
    f = codecs.open('corpus.txt', 'r', 'utf-8')
    corpus = []
    for line in f:
        line = line.strip().split('\t')
        corpus.append(line[1])
    corpus_new = entity_tagging(corpus)
