# -*- coding: utf-8 -*-

# break の使い方

count = 0
while count < 9:
    print 'The count is:', count
    if count == 5:
        print 'I need to break now'
        break
    count = count + 1

print "Good bye!"

# continue の使い方

a = [1, 0, 2, 4]
for element in a:
    if element == 0:
        continue
    print 1. / element

