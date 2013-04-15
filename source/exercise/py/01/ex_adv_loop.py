# -*- coding: utf-8 -*-

# リストを使ったシーケンス

message = "Hello how are you?"

print message.split()

for word in message.split():
    print word

# 反復回数を知るには

words = ('cool', 'powerful', 'readable')

# range() を使うこともできるが
for i in range(0, len(words)):
    print i, words[i]

# enumerate() を使うとよりシンプルに書ける
for index, item in enumerate(words):
    print index, item

# 辞書を使ったループ

d = {'a': 1, 'b':1.2, 'c':1j}

for key, val in d.iteritems():
    print 'Key: %s has value: %s' % (key, val)

