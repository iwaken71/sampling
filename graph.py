# coding: utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sampling as smp
import sys
#グラフデータ
'''
data/email-Enron.txt '\t'
data/com-lj.ungraph.txt '\t'
data/facebook/1912.txt ' '
data/BA10000.txt '\t'
data/roadNet-CA.txt '\t'
data/roadNet-PA.txt '\t'
data/com-amazon.ungraph.txt '\t'
data/com-youtube.ungraph.txt '\t'
data/facebook_combined.txt '\t'
data/as20.txt '\t'
data/p2p-Gnutella04.txt '\t'
data/com-dblp.txt '\t'
'''
#

Filename = 'data/BA10000.txt'
split = '\t'
WriteFilename = "test.txt"
title = "amazon"
NUM = 6
TrueValue = 0.0
def SelectGraph(n):
    global Filename
    global title
    global TrueValue
    if n == 1:
        Filename = 'data/facebook_combined.txt'
        title = "Facebook"
        TrueValue = 0.6055

    elif n == 2:
        Filename = 'data/email-Enron.txt'
        title = "Email"
        TrueValue = 0.497

    elif n == 3:
        Filename = 'data/com-amazon.ungraph.txt'
        title = "Amazon"
        TrueValue = 0.3967

    elif n == 4:
        Filename = 'data/com-lj.ungraph.txt'
        title = "LiveJournal"
        TrueValue = 0.2843
    elif n == 5:
        Filename = 'data/com-dblp.txt'
        title = "DBLP"
        TrueValue = 0.6324
    elif n == 6:
        Filename = 'data/loc-gowalla_edges.txt'
        title = "Gowalla"
        TrueValue = 0.2367
    elif n == 7:
        Filename = 'data/com-youtube.ungraph.txt'
        title = "Youtube"
        TrueValue = 0.0808
    elif n == 8:
        Filename = 'data/p2p-Gnutella04.txt'
        title = "p2p"
        TrueValue = 0.0062175
    elif n == 9:
        Filename = 'data/com-orkut.ungraph.txt'
        title = "Orkut"
        TrueValue = 0.1666
    elif n == 10:
        Filename = 'data/twitter_combined.txt'
        title = "Twitter"
        TrueValue = 0.5653
    elif n == 11:
        Filename = 'data/web-NotreDame.txt'
        title = 'Web'
        TrueValue = 0.23462432072
def main():
    SelectGraph(NUM)
    G = readGraph()


       # G1 = smp.BFS(G,0.01)
#	G = nx.gnp_random_graph(10000,0.2)
#	G = BA_model_Graph(10)
#	G =	nx.powerlaw_cluster_graph(500000,3,0.5)
 #       print(nx.average_clustering(G))
    print("グラフの読み込み終了")
    print(title)
    print(u"真値"+str(TrueValue))
        #plot_CC(G,0.01)
       # print(GCC(G))
        #print(myGCC(G,177820130))
# G2 = MHRW(G,0.1)
#	G1 = smp.Snowball_Sampling(G,0.01)
#	print("2")
#	writeGraph(G,1)
#	Show_Graph(G)
 #       Show_Graph(G1)
#	NDD(smp.RWall(G,0.01))
#       DCDF(G,100)
#        CCCDF(G)
#        CCNMSE(G,5,0.25)
    #NDD(G)
#        G2 = CRW(G,0.25)
#        NDD(G2)
#        CCCDF(G2)
#	print(ans)
#        print("average_shotest_path")
#        print(AD(G))i
#        AD2(G,3,0.01)
       # print(nx.diameter(G))
       # NMSE2(G,100,3,0.01) #誤差計算 G:元のグラフ,最大次数,サンプリング回数,サンプリングの割合
 #       CC2(G,25,0.01)
    lst = []
    error = 0.0
    shinchi = TrueValue
    N =0
        #
#        print(smp.RW(G,0.01))
       # print(Soukan(G))
 #       print(smp.Draw_RW(G,0.01))

    if N > 0:
        for i in range(0,N):
            lst.append(smp.RWall3(G,0.25))
            error += (lst[i]-shinchi)*(lst[i]-shinchi)
          #  lst.append(AD(G1));
      #      lst.append(smp.Estimate(G,0.01))
           # lst.append(smp.RWall3(G,0.01))
            print(lst[i])
  #      for i in range(0,25):
   #         print(lst[i])
        error = error/N
        error = math.sqrt(error)

        Samp(lst)
        print("平均二乗誤差:"+str(error))


    a = 1000
    DaiJikken(G,N=1000,visual=False)
       # NN = 25
       # for i in xrange(NN):
           # smp.Estimate(G,0.01)
    print(0)
       # for i in xrange(NN):
           # smp.Estimate2(G,0.01)
       # nx.draw(G)
      #  plt.show()
    print("finish")
    return 0
def plot_CC(G,p):
    n = int(len(G.nodes())*p)
    x = range(0,n)
    base = []
    a = []
    b = []
    c = []
    d = []

    for i in x:
        base.append(TrueValue)
    '''
    for i in range(25):
        a.append(smp.RW_plot(G,p))
    #    b.append(np.array(smp.MHRW_plot(G,p)))
     #   c.append(np.array(smp.Estimate_plot(G,p,False)))
      #  d.append(np.array(smp.Estimate_plot(G,p,True)))
    a1 = [0]*len(a[0])
    for i in range(25):
        for j in range(0,len(a[0])):
            a1[j] += a[i][j]
    for i in range(len(a1)):
        a1[j] /= 25
    '''
    plt.plot(x,smp.RW_plot(G,p),label='RW')
    plt.plot(x,smp.MHRW_plot(G,p),label='MHRW')
    plt.plot(x,smp.Estimate_plot(G,p,False),label='WWW')
    plt.plot(x,smp.Estimate_plot(G,p,True),label='iwasaki')
    plt.plot(x,base)
    plt.legend()
    plt.show()

def DaiJikken(G,N=1000,visual=False):

    x = []
    mhrw = []
    rw_rw = []
    nbrw_rw = []

    for i in range(0,5):
        p = (i+1)* 2000
        x.append(p)
        print("N="+str(N)+", p="+str(p))
        lst1 = [] #MHRW
        lst2 = [] #Estimate
        lst3 = [] #Estimate_kairyou
        shinchi = TrueValue
        print("MHRW")
        error1 = 0.0
        for i in range(0,N):
            lst1.append(0)#smp.MHRW(G,p))
            if visual:
                print(lst1[i])
            error1 += (lst1[i]-shinchi)*(lst1[i]-shinchi)
        error1 = error1/N
        error1 = math.sqrt(error1)
        error1 = error1/TrueValue
        mhrw.append(error1)
        error2 = 0.0
        print("SRW")
        for i in range(0,N):
            lst2.append(smp.Estimate(G,p))
            error2 += (lst2[i]-shinchi)*(lst2[i]-shinchi)
            if visual:
                print(lst2[i])
        error2 = error2/N
        error2 = math.sqrt(error2)
        error2 = error2/TrueValue
        rw_rw.append(error2)
        print("NBRW")
        error3 = 0.0
        for i in range(0,N):
            lst3.append(smp.Estimate2(G,p))
            error3 += (lst3[i]-shinchi)*(lst3[i]-shinchi)
            if visual:
                print(lst3[i])
        error3 = error3/N
        error3 = math.sqrt(error3)
        error3 = error3/TrueValue
        nbrw_rw.append(error3)
        print("MHRW")
        #Samp(lst1)
        #print("平均二乗誤差:\t"+str(error1))
        print("Estimate")
        Samp(lst2)
        print("平均二乗誤差:\t"+str(error2))
        print("Improve")
        Samp(lst3)
        print("平均二乗誤差:\t"+str(error3))
    x = np.array(x)
    #plt.plot(x,mhrw,label="MHRW-ego",color="black",linestyle="-.")
    plt.plot(x,rw_rw,label="SRW",color="black",linestyle="--")
    plt.plot(x,nbrw_rw,label="NBRW",color="red",linestyle="-")

    plt.xlabel("Sample size",fontsize = 18)
    plt.ylabel("NRMSE",fontsize = 18)
    plt.title(title,fontsize = 18)
    plt.xticks([2000,4000,6000,8000,10000], ["2000","4000","6000","8000","10000"])
    plt.legend(loc = "best")
    plt.show()


def Soukan(G):
    degree = []
    av_degree = 0.0
    cluster = []
    av_cluster = 0.0
    count = 0
    for node in G.nodes():
        count += 1
        d = nx.degree(G,node)
        c = nx.clustering(G,node)
        degree.append(d)
        cluster.append(c)
        av_degree += d
        av_cluster += c
    av_degree /= count
    av_cluster /= count
    print(av_cluster)
    a = 0.0
    b = 0.0
    c = 0.0
    for i in xrange(count):
        a += (degree[i]-av_degree)*(cluster[i]-av_cluster)
        b += (degree[i]-av_degree)**2
        c += (cluster[i]-av_cluster)**2
    b = math.sqrt(b)
    c = math.sqrt(c)
    r = a/(b*c)
    return r


def Jikken(G,N,p,visual = True):
    print("N="+str(N)+", p="+str(p))
    lst1 = [] #RW
    lst2 = [] #MHRW
   # lst3 = [] #iwasaki
    lst4 = [] #Estimate
    lst5 = [] #Estimate_kairyou
    error1 = 0.0
    shinchi = TrueValue
    '''
    print("RW")
    for i in range(0,N):
        lst1.append(smp.RWRW(G,p))
        error1 += (lst1[i]-shinchi)*(lst1[i]-shinchi)
        if visual:
            print(lst1[i])
    error1 = error1/N
    error1 = math.sqrt(error1)
    '''
    print("MHRW")
    error2 = 0.0
    for i in range(0,N):
    #    sys.stdout.write("|")
        lst2.append(1)#smp.MHRW(G,p))
        error2 += (lst2[i]-shinchi)*(lst2[i]-shinchi)
        if visual:
            print(lst2[i])
    error2 = error2/N
    error2 = math.sqrt(error2)
    #print("iwasaki")
   # error3 = 0.0
   # for i in range(0,N):
    #    lst3.append(smp.RWall3(G,p))
    #    error3 += (lst3[i]-shinchi)*(lst3[i]-shinchi)
     #   print(lst3[i])
   # error3 = error3/N
   # error3 = math.sqrt(error3)
    error4 = 0.0
    print("Estimate")
    for i in range(0,N):
        lst4.append(smp.Estimate(G,p))
        error4 += (lst4[i]-shinchi)*(lst4[i]-shinchi)
        if visual:
            print(lst4[i])
    error4 = error4/N
    error4 = math.sqrt(error4)
    print("improve")
    error5 = 0.0
    for i in range(0,N):
        lst5.append(smp.Estimate2(G,p))
        error5 += (lst5[i]-shinchi)*(lst5[i]-shinchi)
        if visual:
            print(lst5[i])
    error5 = error5/N
    error5 = math.sqrt(error5)
    #print("RW")
    #Samp(lst1)
   # print("平均二乗誤差:"+str(error1))
    print("MHRW")
    Samp(lst2)
    print("平均二乗誤差:\t"+str(error2))
   # print("iwasaki")
   # Samp(lst3)
   # print("平均二乗誤差:"+str(error3))
    print("Estimate")
    Samp(lst4)
    print("平均二乗誤差:\t"+str(error4))
    print("Improve")
    Samp(lst5)
    print("平均二乗誤差:\t"+str(error5))

def readGraph():
    G = nx.Graph()
    f = open(Filename,"r")
    for line in f:
        args = line[:-1].split('\t')
        if not (args[0] == args[1]):
            G.add_edge(args[0],args[1])
    return G

	#G = nx.read_adjlist(Filename,delimiter=split,create_using =nx.Graph())
	#return G

def writeGraph(G,n):
        if n == 1:

	    f = open(title+"_BFS.txt", 'w')
	    for pair in G.edges():
	    	    f.write(str(pair[0]))
		    f.write('\t')
		    f.write(str(pair[1]))
		    f.write('\n')
	    f.close()
        elif n == 2:

	    f = open(title+"_MHRW.txt", 'w')
	    for pair in G.edges():
	    	    f.write(str(pair[0]))
		    f.write('\t')
		    f.write(str(pair[1]))
		    f.write('\n')
	    f.close()
        elif n == 3:

            f = open(title+"_BAS.txt", 'w')
	    for pair in G.edges():
	    	    f.write(str(pair[0]))
		    f.write('\t')
		    f.write(str(pair[1]))
		    f.write('\n')
            f.close()

def GCC(G):
    return nx.transitivity(G)
def myGCC(G,tri):

    bunbo = 0.0
    for node in G.nodes():
        d = G.degree(node)
        if d >= 2:
            bunbo += d*(d-1)

    return 6.0*tri/bunbo


def AverageDegree(G):
    return 2.0*nx.number_of_edges(G)/nx.number_of_nodes(G)

#引数Gのグラフの次数分布を表示する
def NDD(G):
	degree_sequence = sorted(nx.degree(G).values(),reverse = True)
#	print ("Degree sequence", degree_sequence)
	dmax = max(degree_sequence)
#	print(dmax)
	kukan=range(0,dmax+4)
	hist, kukan = np.histogram(degree_sequence,kukan,normed=True)
#	print(hist,kukan)
	plt.yscale('log')
	plt.xscale('log')
	plt.plot(hist)
	plt.xlabel('degree')
	plt.ylabel('frequency')
	plt.grid(True)
	plt.show()
	return 0
def NDD2(G,p):
    	degree_sequence = sorted(nx.degree(G).values(),reverse = True)
#	print ("Degree sequence", degree_sequence)
	dmax = max(degree_sequence)
#	print(dmax)
	kukan=range(0,dmax+4)
	hist, kukan = np.histogram(degree_sequence,kukan,normed=True)
#	print(hist,kukan)
	plt.figure(2)
	plt.yscale('log')
	plt.xscale('log')

        G1 = smp.BFS(G,p)
        degree_sequence = sorted(nx.degree(G1).values(),reverse = True)
#	print ("Degree sequence", degree_sequence)
#	print(dmax)
	hist1 ,kukan1 = np.histogram(degree_sequence,kukan,normed=True)
#	print(hist,kukan)

        G2 = smp.MHRW(G,p)
        d = sorted(nx.degree(G2).values(),reverse = True)
        hist2 ,kukan2 = np.histogram(d,kukan,normed=True)

        G3 = smp.BAS(G,p)
        d = sorted(nx.degree(G3).values(),reverse = True)
        hist3 ,kukan3 = np.histogram(d,kukan,normed=True)



	plt.plot(hist1,label="BFS")
        plt.plot(hist2,label="MHRW")
        plt.plot(hist3,label=u"提案手法")
        plt.plot(hist,label=u"真値")
       # plt.plot(hist,label="Facebook",color="black",linewidth=1.5)
        plt.legend()
        plt.title("NDD")
        plt.xlabel('degree')
        plt.ylabel('frequency')
	plt.grid(True)
	plt.show()
	return 0


#平均距離を調べる
def AD2(G,n,p):
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sample1 = []
   # normal = AD(G)
   # print(title+":"+str(normal))
  #  f = open(title+"_BAS_CC.txt", 'w')
    for i in range(0,n):
       # G1 = smp.BFS(G,p)
        G1 = smp.RWall2(G,p)
       # G2 = smp.MHRW(G,p)
       # G3 = smp.BAS(G,p)

        val1 = AD(G1)
        sample1.append(val1)
       # sum2 += CC(G2)
       # val3 = CC(G3)
   #     f.write(str(val1))
    ##    f.write('\n')
        print(val1)
       # print(val3)
        sum1 += val1
    #    sum3 += val3

    #sum1 = sum1/n
   # sum2 = sum2/n
   # sum3 = sum3/n
   # f.write("BAS_CC_av"+str(sum3))
  #  f.close()

   # print("RWall:"+str(sum1))
   # print("MHRW:"+str(sum2))
   # print("BAS:"+str(sum3))
    sum1 = 0
    for x in sample1:
        sum1 += x
    ave1 = sum1/len(sample1)

    var1 = 0
    for x in sample1:
        var1 += (x - ave1)*(x - ave1)
    var1 = var1/len(sample1)
    hensa1 = math.sqrt(var1)
    left = ave1 - 1.96 * hensa1 / (math.sqrt(len(sample1)))
    right = ave1 + 1.96 * hensa1 / (math.sqrt(len(sample1)))
    print("平均:"+str(ave1))
    print("標準偏差:"+str(hensa1))
    print("信頼区間95%"+str(left)+"~"+str(right))

#クラスタ係数を調べる
def GCC2(G,n,p):
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sample1 = []
   # f = open(title+"_BAS_CC.txt", 'w')
    for i in range(0,n):
       # G1 = smp.BFS(G,p)
        G1 = smp.MHRW(G,p)
       # G2 = smp.MHRW(G,p)
       # G3 = smp.BAS(G,p)

        val1 = GCC(G1)
        sample1.append(val1)
       # sum2 += CC(G2)
       # val3 = CC(G3)
       # f.write(str(val1))
       # f.write('\n')
        print(val1)
       # print(val3)
        sum1 += val1
    #    sum3 += val3

    sum1 = sum1/n
   # sum2 = sum2/n
   # sum3 = sum3/n
   # f.write("BAS_CC_av"+str(sum3))
   # f.close()

    print("RWall:"+str(sum1))
   # print("MHRW:"+str(sum2))
   # print("BAS:"+str(sum3))
    sum1 = 0
    for x in sample1:
        sum1 += x
    ave1 = sum1/len(sample1)

    var1 = 0
    for x in sample1:
        var1 += (x - ave1)*(x - ave1)
    var1 = var1/len(sample1)
    hensa1 = math.sqrt(var1)
    left = ave1 - 1.96 * hensa1 / (math.sqrt(len(sample1)))
    right = ave1 + 1.96 * hensa1 / (math.sqrt(len(sample1)))
    print("平均:"+str(ave1))
    print("標準偏差:"+str(hensa1))
    print("信頼区間95%"+str(left)+"~"+str(right))

def CC2(G,n,p):
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sample1 = []
    error = []
   # f = open(title+"_BAS_CC.txt", 'w')
    for i in range(0,n):
       # G1 = smp.BFS(G,p)
        G1 = smp.Snowball_Sampling(G,p)
       # G2 = smp.MHRW(G,p)
       # G3 = smp.BAS(G,p)

        val1 = CC(G1)
        sample1.append(val1)
       # sum2 += CC(G2)
       # val3 = CC(G3)
    #    f.write(str(val1))
     #   f.write('\n')
        print(val1)
#        error.append((((float)val1-(float)TrueValue))^2)
       # print(val3)
        sum1 += val1
    #    sum3 += val3

    sum1 = sum1/n
   # sum2 = sum2/n
   # sum3 = sum3/n
   # f.write("BAS_CC_av"+str(sum3))
   # f.close()

    print("RWall:"+str(sum1))
   # print("MHRW:"+str(sum2))
   # print("BAS:"+str(sum3))
    sum1 = 0
    for x in sample1:
        sum1 += x
    ave1 = sum1/len(sample1)
    sum2 = 0
    for x in error:
        sum2 += x
#    ave2 = sum2/(len(error))

    var1 = 0
    for x in sample1:
        var1 += (x - ave1)*(x - ave1)
    var1 = var1/len(sample1)
    hensa1 = math.sqrt(var1)
    left = ave1 - 1.96 * hensa1 / (math.sqrt(len(sample1)))
    right = ave1 + 1.96 * hensa1 / (math.sqrt(len(sample1)))
    print("平均:"+str(ave1))
    print("標準偏差:"+str(hensa1))
    print("信頼区間95%"+str(left)+"~"+str(right))
 #   print("誤差"+str(ave2))

def Samp(lst):
    sum1 = 0.0
    for x in lst:
        sum1 += x
    ave1 = sum1/len(lst)
    var1 = 0.0
    for x in lst:
        var1 += (x-ave1)*(x-ave1)
    var1 = var1/len(lst)
    hensa1 = math.sqrt(var1)
    left = ave1 - 1.96 * hensa1/(math.sqrt(len(lst)))
    right = ave1 + 1.96 * hensa1/(math.sqrt(len(lst)))
    print("平均:\t"+str(ave1))
    print("標準偏差:\t"+str(hensa1))
    print("信頼区間95%\t"+str(left)+"\t"+str(right))


def DCDF(G,max_k):

	degree_sequence0 = sorted(nx.degree(G).values()) #次数
	size0 = len(degree_sequence0) #頂点数
	dmax0 = max(degree_sequence0) #次数の最大
	list0 = [1.0]*(max_k+1) #計算に使うリスト
	pivot0 = 0
	finish = False
	for i in range(0,size0):
		while True:
			if degree_sequence0[i] == pivot0:
				if i == size0-1:
					list0[pivot0] = 1.0
				break #次のiに進む
			else:
				if pivot0 >= dmax0:
					list0[pivot0] = 1.0
				else:
					list0[pivot0] = i/float(size0)
				pivot0 += 1
				if pivot0 > max_k:
					finish = True
					break

		if finish:
			break

        plt.plot(list0)
        plt.show()

def CC(G):
	return nx.average_clustering(G)
def AD(G):
        return nx.average_shortest_path_length(G)

def CCCDF(G):
	C_sequence0 = sorted(nx.clustering(G).values()) #次数
	size0 = len(C_sequence0) #頂点数
	Cmax0 = max(C_sequence0) #次数の最大
        x = np.arange(0, 1.01, 0.01)
	list0 = [1.0]*(len(x)) #計算に使うリスト
#	print(len(x))
        pivot0 = 0
	finish = False
	for i in range(0,size0):
		while True:
			if C_sequence0[i] <= pivot0*0.01:
				if i == size0-1:
					list0[pivot0] = 1.0
				break #次のiに進む
			else:
				if pivot0*0.01 >= Cmax0:
					list0[pivot0] = 1.0
				else:
					list0[pivot0] = i/float(size0)
				pivot0 += 1
				if pivot0 >= len(x):
					finish = True
					break

		if finish:
			break
      # x = np.arange(0, 0.5, 0.01)
        plt.plot(x,list0)
        plt.show()

#サンプリングの誤差を調べる
def NMSE2(G,max_k,num,p):
	# max_k: 調べる次数の上限
	# num = 10 # サンプルする回数
	print(1)

	degree_sequence0 = sorted(nx.degree(G).values()) #次数
	size0 = len(degree_sequence0) #頂点数
	dmax0 = max(degree_sequence0) #次数の最大
        list0 = [1.0]*(max_k+1) #計算に使うリスト
	pivot0 = 0
	finish = False
	for i in range(0,size0):
		while True:
			if degree_sequence0[i] == pivot0:
				if i == size0-1:
					list0[pivot0] = 1.0
				break #次のiに進む
			else:
				if pivot0 >= dmax0:
					list0[pivot0] = 1.0
				else:
					list0[pivot0] = i/float(size0)
				pivot0 += 1
				if pivot0 > max_k:
					finish = True
					break

		if finish:
			break
#	print(list0)
    	C_sequence0 = sorted(nx.clustering(G).values()) #次数
	size0 = len(C_sequence0) #頂点数
	Cmax0 = max(C_sequence0) #次数の最大
        cx = np.arange(0, 1.01, 0.01)
	Clist = [1.0]*(len(cx)) #計算に使うリスト

	finish = False
	for i in range(0,size0):
		while True:
			if C_sequence0[i] <= pivot0*0.01:
				if i == size0-1:
				        Clist[pivot0] = 1.0
				break #次のiに進む
			else:
				if pivot0*0.01 >= Cmax0:
					Clist[pivot0] = 1.0
				else:
					Clist[pivot0] = i/float(size0)
				pivot0 += 1
				if pivot0 >= len(cx):
					finish = True
					break

		if finish:
			break

	map1 = {}
	map2 = {}
        map3 = {}
        Cmap1 = {}
        Cmap2 = {}
        Cmap3 = {}
	for j in range(0,num):
		G1 = smp.BFS(G,p)
               # G1 = smp.RWall(G,p)
                if j == 0:
                        writeGraph(G1,1)
		degree_sequence1 = sorted(nx.degree(G1).values())
		size1 = len(degree_sequence1)
		dmax1 = max(degree_sequence1)
		list1 = [1.0]*(max_k+1)
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if degree_sequence1[i] == k1:
					if i == size1-1:
						list1[k1] = 1.0
					break
                		else:
					if k1 >= dmax1:
						list1[k1] = 1.0
					else:
						list1[k1] = i/float(size1)
					k1 = k1 + 1
					if k1 > max_k:
						finish = True
						break
			if finish:
				break
		map1[j] = list1
        	C_sequence1 = sorted(nx.clustering(G1).values())
                size1 = len(C_sequence1)
		Cmax1 = max(C_sequence1)
		Clist1 = [1.0]*(len(cx))
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if C_sequence1[i] <= k1*0.01:
					if i == size1-1:
						Clist1[k1] = 1.0
					break
				else:
					if k1*0.01 >= Cmax1:
						Clist1[k1] = 1.0
					else:
						Clist1[k1] = i/float(size1)
					k1 += 1
					if k1 >= len(cx):
						finish = True
						break
			if finish:
				break
		Cmap1[j] = Clist1

	for j in range(0,num):
		G1 = smp.MHRW(G,p)
                if j == 0:
                        writeGraph(G1,2)
                degree_sequence1 = sorted(nx.degree(G1).values())
		size1 = len(degree_sequence1)
		dmax1 = max(degree_sequence1)
		list1 = [1.0]*(max_k+1)
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if degree_sequence1[i] == k1:
					if i == size1-1:
						list1[k1] = 1.0
					break
				else:
					if k1 >= dmax1:
						list1[k1] = 1.0
					else:
						list1[k1] = i/float(size1)
					k1 = k1 + 1
					if k1 > max_k:
						finish = True
						break
			if finish:
				break
		map2[j] = list1

        	C_sequence1 = sorted(nx.clustering(G1).values())
		size1 = len(C_sequence1)
		Cmax1 = max(C_sequence1)
		Clist1 = [1.0]*(len(cx))
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if C_sequence1[i] <= k1*0.01:
					if i == size1-1:
						Clist1[k1] = 1.0
					break
				else:
					if k1*0.01 >= Cmax1:
						Clist1[k1] = 1.0
					else:
						Clist1[k1] = i/float(size1)
					k1 += 1
					if k1 >= len(cx):
						finish = True
						break
			if finish:
				break
		Cmap2[j] = Clist1
	for j in range(0,num):
		G1 = smp.RWall2(G,p)
                if j == 0:
                        writeGraph(G1,3)
		degree_sequence1 = sorted(nx.degree(G1).values())
		size1 = len(degree_sequence1)
		dmax1 = max(degree_sequence1)
		list1 = [1.0]*(max_k+1)
		k1 = 0
		finish = False
		for i in range(0,size1):
                        while True:
				if degree_sequence1[i] == k1:
					if i == size1-1:
						list1[k1] = 1.0
					break
				else:
					if k1 >= dmax1:
						list1[k1] = 1.0
					else:
						list1[k1] = i/float(size1)
					k1 = k1 + 1
					if k1 > max_k:
			        		finish = True
						break
			if finish:
				break
		map3[j] = list1
		C_sequence1 = sorted(nx.clustering(G1).values())
		size1 = len(C_sequence1)
		Cmax1 = max(C_sequence1)
		Clist1 = [1.0]*(len(cx))
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if C_sequence1[i] <= k1*0.01:
					if i == size1-1:
						Clist1[k1] = 1.0
					break
				else:
					if k1*0.01 >= Cmax1:
						Clist1[k1] = 1.0
					else:
						Clist1[k1] = i/float(size1)
					k1 += 1
					if k1 >= len(cx):
						finish = True
						break
			if finish:
				break
		Cmap3[j] = Clist1


	ans = [0]*(max_k+1)
	ans2 = [0]*(max_k+1)
        ans3 = [0]*(max_k+1)
	for i in range(0,max_k+1):
		sum = 0.0
		value0 = list0[i]

		for j in range(0,num):
			list2 = map1[j]
			#print(list2)
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			ans[i] = 0
		else:
			ans[i] = math.sqrt(sum)/value0


	for i in range(0,max_k+1):
		sum = 0.0
		value0 = list0[i]

		for j in range(0,num):
			list2 = map2[j]
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			ans2[i] = 0
		else:
	                ans2[i] = math.sqrt(sum)/value0

        for i in range(0,max_k+1):
		sum = 0.0
		value0 = list0[i]

		for j in range(0,num):
			list2 = map3[j]
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			ans3[i] = 0
		else:
			ans3[i] = math.sqrt(sum)/value0



	plt.plot(ans,label="BFS")
	plt.plot(ans2,label="MHRW")
        plt.plot(ans3,label=u"提案手法")
	plt.legend()
        plt.title(title)
        plt.xlabel("degree")
        plt.ylabel("NMSE")

        plt.show()

	Cans = [0]*(len(cx))
	Cans2 = [0]*(len(cx))
	Cans3 = [0]*(len(cx))
	for i in range(0,len(cx)):
		sum = 0.0
		value0 = Clist[i]

		for j in range(0,num):
			list2 = Cmap1[j]
                        sum += ((value0-list2[i])**2)

                sum = sum/num
                if value0 == 0:
                        Cans[i] = 0
                else:
                        Cans[i] = math.sqrt(sum)/value0
	for i in range(0,len(cx)):
		sum = 0.0
		value0 = Clist[i]

		for j in range(0,num):
			list2 = Cmap2[j]
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			Cans2[i] = 0
		else:
			Cans2[i] = math.sqrt(sum)/value0
	for i in range(0,len(cx)):
		sum = 0.0
		value0 = Clist[i]

		for j in range(0,num):
			list2 = Cmap3[j]
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			Cans3[i] = 0
		else:
			Cans3[i] = math.sqrt(sum)/value0



	plt.plot(cx,Cans,label="RWall")
	plt.plot(cx,Cans2,label="MHRW")
	plt.plot(cx,Cans3,label=u"提案手法")
        plt.legend()
        plt.title(title)
        plt.xlabel("Clustering Coefficient")
        plt.ylabel("NMSE")
	plt.show()

	return 0


def CCNMSE(G,num,p):
	C_sequence0 = sorted(nx.clustering(G).values()) #次数
	size0 = len(C_sequence0) #頂点数
	Cmax0 = max(C_sequence0) #次数の最大
        cx = np.arange(0, 1.01, 0.01)
	Clist = [1.0]*(len(cx)) #計算に使うリスト
	pivot0 = 0
	finish = False
	for i in range(0,size0):
		while True:
			if C_sequence0[i] <= pivot0*0.01:
				if i == size0-1:
					Clist[pivot0] = 1.0
				break #次のiに進む
			else:
				if pivot0*0.01 >= Cmax0:
					Clist[pivot0] = 1.0
				else:
					Clist[pivot0] = i/float(size0)
				pivot0 += 1
				if pivot0 >= len(cx):
					finish = True
					break

		if finish:
			break
	Cmap1 = {}
	Cmap2 = {}
        Cmap3 = {}

	for j in range(0,num):
		G1 = smp.BFS(G,p)
		C_sequence1 = sorted(nx.clustering(G1).values())
		size1 = len(C_sequence1)
		Cmax1 = max(C_sequence1)
		Clist1 = [1.0]*(len(cx))
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if C_sequence1[i] <= k1*0.01:
					if i == size1-1:
						Clist1[k1] = 1.0
					break
				else:
					if k1*0.01 >= Cmax1:
						Clist1[k1] = 1.0
					else:
						Clist1[k1] = i/float(size1)
					k1 += 1
					if k1 >= len(cx):
						finish = True
						break
			if finish:
				break
		Cmap1[j] = Clist1
	for j in range(0,num):
		G1 = smp.MHRW(G,p)
		C_sequence1 = sorted(nx.clustering(G1).values())
		size1 = len(C_sequence1)
		Cmax1 = max(C_sequence1)
		Clist1 = [1.0]*(len(cx))
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if C_sequence1[i] <= k1*0.01:
					if i == size1-1:
						Clist1[k1] = 1.0
					break
				else:
					if k1*0.01 >= Cmax1:
						Clist1[k1] = 1.0
					else:
						Clist1[k1] = i/float(size1)
					k1 += 1
					if k1 >= len(cx):
						finish = True
						break
			if finish:
				break
		Cmap2[j] = Clist1
	for j in range(0,num):
		G1 = smp.BAS(G,p)
		C_sequence1 = sorted(nx.clustering(G1).values())
		size1 = len(C_sequence1)
                Cmax1 = max(C_sequence1)
                Clist1 = [1.0]*(len(cx))
                k1 = 0
                finish = False
                for i in range(0,size1):
			while True:
				if C_sequence1[i] <= k1*0.01:
					if i == size1-1:
						Clist1[k1] = 1.0
					break
				else:
					if k1*0.01 >= Cmax1:
						Clist1[k1] = 1.0
					else:
						Clist1[k1] = i/float(size1)
					k1 += 1
					if k1 >= len(cx):
						finish = True
						break
			if finish:
				break
		Cmap3[j] = Clist1


	Cans = [0]*(len(cx))
	Cans2 = [0]*(len(cx))
	Cans3 = [0]*(len(cx))
	for i in range(0,len(cx)):
		sum = 0.0
		value0 = Clist[i]

		for j in range(0,num):
			list2 = Cmap1[j]
                        sum += ((value0-list2[i])**2)

                sum = sum/num
                if value0 == 0:
                        Cans[i] = 0
                else:
                        Cans[i] = math.sqrt(sum)/value0
	for i in range(0,len(cx)):
		sum = 0.0
		value0 = Clist[i]

		for j in range(0,num):
			list2 = Cmap2[j]
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			Cans2[i] = 0
		else:
			Cans2[i] = math.sqrt(sum)/value0
	for i in range(0,len(cx)):
		sum = 0.0
		value0 = Clist[i]

		for j in range(0,num):
			list2 = Cmap3[j]
			sum += ((value0 - list2[i])**2)

		sum = sum/num
		if value0 == 0:
			Cans3[i] = 0
		else:
			Cans3[i] = math.sqrt(sum)/value0



	plt.plot(cx,Cans,label="BFS")
	plt.plot(cx,Cans2,label="MHRW")
	plt.plot(cx,Cans3,label=u"提案手法")
        plt.legend()
	plt.show()

	return 0

def Complete_Graph(n):
    assert n > 0, "n is num nodes"
    G1 = nx.Graph()
    #G1.add_nodes_from(range(0,n))

    for i in range(0,n-1):
        for j in range(i+1,n):
            G1.add_edge(i,j)
    return G1

def BA_model_Graph(n):
    assert n > 0, "n is num nodes"

    node_count = 0
    complete_number = 3
    SUM = complete_number*2
    G = Complete_Graph(complete_number)
    node_count += complete_number
    degree = []
    for i in range(0,complete_number):
        degree.append(complete_number-1)
  #  print("test")

    while node_count < n:
        G.add_node(node_count)

        count = 0
        for i in range(0,node_count):
            if random.randint(1,SUM) <= degree[i]:
                G.add_edge(i,node_count)
                degree[i] = degree[i] + 1
                count += 1
        if count == 0:
            G.remove_node(node_count)
        else:
            degree.append(count)
            node_count += 1
            SUM += count*2
    return G

def Show_Graph(G):
	nx.draw(G)
	plt.show()

def NMSE(G0,max_k,num = 100,p = 0.1):
	# max_k: 調べる次数の上限
	# num # サンプルする回数
	print(1)

	degree_sequence0 = sorted(nx.degree(G0).values()) #次数
	size0 = len(degree_sequence0) #頂点数
	dmax0 = max(degree_sequence0) #次数の最大
	list0 = [1.0]*(max_k+1) #計算に使うリスト
	pivot0 = 0
	finish = False
	for i in range(0,size0):
		while True:
			if degree_sequence0[i] == pivot0:
				if i == size0-1:
					list0[pivot0] = 1.0
				break #次のiに進む
			else:
				if pivot0 >= dmax0:
					list0[pivot0] = 1.0
				else:
					list0[pivot0] = i/float(size0)
				pivot0 += 1
				if pivot0 > max_k:
					finish = True
					break

		if finish:
			break
#	print(list0)

	map1 = {}
	map2 = {}
        map3 = {}
	for j in range(0,num):
		G1 = smp.BFS(G0,p)
		degree_sequence1 = sorted(nx.degree(G1).values())
		size1 = len(degree_sequence1)
		dmax1 = max(degree_sequence1)
		list1 = [1.0]*(max_k+1)
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if degree_sequence1[i] == k1:
					if i == size1-1:
						list1[k1] = 1.0
					break
				else:
					if k1 >= dmax1:
						list1[k1] = 1.0
					else:
						list1[k1] = i/float(size1)
					k1 = k1 + 1
					if k1 > max_k:
						finish = True
						break
			if finish:
				break
		map1[j] = list1
                #print(list1)
	for j in range(0,num):
		G1 = smp.MHRW(G0,p)
		degree_sequence1 = sorted(nx.degree(G1).values())
		size1 = len(degree_sequence1)
		dmax1 = max(degree_sequence1)
		list1 = [1.0]*(max_k+1)
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if degree_sequence1[i] == k1:
					if i == size1-1:
						list1[k1] = 1.0
					break
				else:
					if k1 >= dmax1:
						list1[k1] = 1.0
					else:
						list1[k1] = i/float(size1)
                                        k1 = k1 + 1
					if k1 > max_k:
						finish = True
						break
			if finish:
				break
		map2[j] = list1
                #print(list1)


	for j in range(0,num):
		G1 = smp.BAS(G0,p)
		degree_sequence1 = sorted(nx.degree(G1).values())
		size1 = len(degree_sequence1)
		dmax1 = max(degree_sequence1)
		list1 = [1.0]*(max_k+1)
		k1 = 0
		finish = False
		for i in range(0,size1):
			while True:
				if degree_sequence1[i] == k1:
					if i == size1-1:
						list1[k1] = 1.0
					break
				else:
					if k1 >= dmax1:
						list1[k1] = 1.0
					else:
						list1[k1] = i/float(size1)
					k1 = k1 + 1
					if k1 > max_k:
						finish = True
						break
			if finish:
				break
		map3[j] = list1
                # print(list1)

	ans = [0]*(max_k+1)
	ans2 = [0]*(max_k+1)
        ans3 = [0]*(max_k+1)
	for i in range(0,max_k+1):
		sum1 = 0.0
                sum2 = 0.0
                sum3 = 0.0
		value0 = list0[i]

		for j in range(0,num):
			list1 = map1[j]
                        list2 = map2[j]
                        list3 = map3[j]
			#print(list2)
			sum1 += ((value0 - list1[i])**2)
                        sum2 += ((value0 - list2[i])**2)
                        sum3 += ((value0 - list3[i])**2)

		sum1 = sum1/num
                sum2 = sum2/num
                sum3 = sum3/num
               # print(sum1)
               # print(value0)
		if value0 == 0:
			ans[i] = 0
                        ans2[i] = 0
                        ans3[i] = 0
		else:
			ans[i] = math.sqrt(sum1)/value0
                        ans2[i] = math.sqrt(sum2)/value0
                        ans3[i] = math.sqrt(sum3)/value0
        #
        '''

	for i in range(0,max_k+1):
		sum1 = 0.0
		value0 = list0[i]

		for j in range(0,num):
			list2 = map2[j]
			sum1 += ((value0 - list2[i])**2)

		sum1 = sum1/num
		if value0 == 0:
			ans2[i] = 0
		else:
	                ans2[i] = math.sqrt(sum1)/value0

        for i in range(0,max_k+1):
		sum1 = 0.0
		value0 = list0[i]

		for j in range(0,num):
			list2 = map3[j]
			sum1 += ((value0 - list2[i])**2)

		sum1 = sum1/num
		if value0 == 0:
			ans3[i] = 0
		else:
			ans3[i] = math.sqrt(sum1)/value0
	'''


	plt.plot(ans,label="BFS")
	plt.plot(ans2,label="MHRW")
        plt.plot(ans3,label=u"提案手法")
	plt.legend()
	plt.show()

	return 0


if __name__ == "__main__":
    main()
