import re
import math

f = open('out.txt', 'r')

regex = re.compile('\s*(?P<id>\d+)\s+\d+:Class_(?P<actual>\d+)\s+\d+:Class_(?P<predicted>\d+)\s+(?P<error>\+?)\s+(?P<a1>\*?)(?P<p1>[01](\.\d*)?),(?P<a2>\*?)(?P<p2>[01](\.\d*)?),(?P<a3>\*?)(?P<p3>[01](\.\d*)?),(?P<a4>\*?)(?P<p4>[01](\.\d*)?),(?P<a5>\*?)(?P<p5>[01](\.\d*)?),(?P<a6>\*?)(?P<p6>[01](\.\d*)?),(?P<a7>\*?)(?P<p7>[01](\.\d*)?),(?P<a8>\*?)(?P<p8>[01](\.\d*)?),(?P<a9>\*?)(?P<p9>[01](\.\d*)?)')

loss = 0
n = 0
l = 0
for line in f:
    m = regex.match(line)
    actual = int(m.group('actual')) - 1
    predicted = int(m.group('predicted')) - 1
    distr = [float(m.group('p1')),float(m.group('p2')),float(m.group('p3')),float(m.group('p4')),float(m.group('p5')),float(m.group('p6')),float(m.group('p7')),float(m.group('p8')),float(m.group('p9'))]
    if distr[actual] > 0:
        loss -= math.log(distr[actual])
        n = n + 1
    else:
        l = l + 1

print n
print l
print loss/n
