import csv
import pprint
import matplotlib.pyplot as plt
import math

path = '/Users/yutaro/research/2020/src/build/'

with open(path+'ground_truth.csv') as f:
   #print(f.read())
   readf = list(csv.reader(f))
   with open(path+'resultZ=0.1115.csv') as g:
        readg = list(csv.reader(g))

        l = []
        lr = []
        x = []
        N = 30

        for i in range(2,17):
            x.append((i+7) *180./ N)

            diff = 0
            r = 0

            for j in range(1,4):
                diff += math.sqrt((float(readf[i][j] )- float(readg[i][j])) *  (float(readf[i][j]) - float(readg[i][j])))
                r += float(readf[i][j]) * float(readf[i][j])


            r = math.sqrt(r)


            l.append(diff)
            lr.append(diff/r)


fig = plt.figure()
plt.plot(x,l)
plt.show()
plt.savefig(path+"diff.png")
fig = plt.figure()
plt.plot(x,lr)
plt.show()
plt.savefig(path+"diff_r.png")
            

