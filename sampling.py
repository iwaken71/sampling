# coding: utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import graph

class Blist:
    def __init__(self):
        self.lst = []
        self.size = 0
        self.map1 = {}
    def insert(self,start,goal,node):
        size = self.lst.count
        if start > goal:
            self.lst.append(node)
            self.put(node)
            return 0

        if size <10:
            for i in range(0,size):
                if node <= self.lst[i]:
                    self.lst.insert(i,node)
                    self.put(node)
                    break
                if i == size-1:
                    self.lst.append(node)
                    self.put(node)
            return 0
        num = (start+goal)/2
        tmp = self.lst[num]
        if start == goal:
            if node <= tmp:
                self.lst.insert(num,node)
                self.put(node)
                return 0
            if tmp < node:
                if num == size-1:
                    self.lst.append(node)
                    self.put(node)
                    return 0
                self.lst.insert(num+1,node)
                self.put(node)
                return 0
        if tmp == node:
            self.lst.insert(num,node)
            self.put(node)
            return 0
        if node < tmp:
            self.insert(start,num-1,node)
        if tmp < node:
            self.insert(num+1,goal,node)
    
    def select(self):
        size = len(self.lst)
        ran = random.randint(0,size-1)
        return self.lst[ran]

    def remove(self,start,goal,node):
        if not self.map1.has_key(node):
            return 0
        num = (start+goal)/2 #真ん中を探す
        tmp = self.lst[num]
        size = self.lst.count
        if tmp == node:
            self.lst.pop(num)
            num2 = self.minus(node)
            if num2 == 0:
                return 0
            
            self.remove(0,size-1,node)
        if node < tmp:
            self.remove(start,num-1,node)
        if tmp < node:
            self.remove(num+1,goal,node)
        return 0   


    def put(self,node):
       # self.size += 1
        if self.map1.has_key(node):
            self.map1[node] += 1
        else:
            self.map1[node] = 1
        return 0
    def minus(self,node):
       # self.size -= 1
        if self.map1.has_key(node):
            if self.map1[node] == 1:
                del self.map1[node]
                return 0
            else:
                self.map1[node] -= 1
                return self.map1[node]
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

def BFS(G,p):
	if p > 1:
		p = 1
	n = int(p*nx.number_of_nodes(G))
       # print(n)
        G1 = nx.Graph()
        check = False
        while nx.number_of_nodes(G1) < n:
                process = []
                start_id = random.randint(0,nx.number_of_nodes(G)-1)
                s_node = nx.nodes(G)[start_id]
                process.append(s_node)
               ######### print(process)
                while True:
                        now_node = process[0]
                        neighbor = G.neighbors(now_node)
                        for next_node in neighbor:
                                if next_node not in G1.nodes():
                                        G1.add_node(now_node)
                                        G1.add_node(next_node)
                                        process.append(next_node)
                                else:
                                        G1.add_node(now_node)
                                        G1.add_node(next_node)
 
                        process.remove(now_node)
                        if nx.number_of_nodes(G1)>=n:
                                check = True
                                break
                        if len(process) == 0:
                                break
                if check:
                        break
        return NodeConnect(G,G1)
            
def MHRW(G,p):
	if p > 1:
		p = 1
	n = int(p*nx.number_of_nodes(G))
       # print(n)
        G1 = nx.Graph()
	d = {}
	def Q(G,v):
		if d.has_key(v):
			return d[v]
		else:
			ans = float(len(G.neighbors(v)))
			d[v] = ans
		return ans

    # a non-zero degree seed is selected at random
	while nx.number_of_nodes(G1) < n:
	        count = 0
	        tmp = 0
                while tmp <= 0:
                        start_id = random.randint(0,nx.number_of_nodes(G)-1)
                        s_node = nx.nodes(G)[start_id]
                        tmp = len(G.neighbors(s_node))

                now_node = s_node
                while count < 5*n:

                        neighber_list = G.neighbors(now_node)
                        next_id = random.randint(0,len(neighber_list)-1)
                        next_node = neighber_list[next_id]
                        p = random.random()
                        value = Q(G,now_node)/Q(G,next_node)

                        if p <= value:

                               # G1.add_edge(now_node,next_node)
                                G1.add_node(now_node)
                               # print(next_node)
                                now_node = next_node
                                count += 1
                        if(nx.number_of_nodes(G1) >= n):
                                return NodeConnect(G,G1)
	return NodeConnect(G,G1)
			   
def RW(G,p):
	if p > 1:
		p = 1
	n = int(p*nx.number_of_nodes(G))
	G1 = nx.Graph()
	while nx.number_of_nodes(G1) < n:
		count = 0
		start_id = random.randint(0,nx.number_of_nodes(G)-1)
		s_node = nx.nodes(G)[start_id]
		now_node = s_node
		while count < 5*n:
            #print(now_node)
			neighber_list = G.neighbors(now_node)
			next_id = random.randint(0,len(neighber_list)-1)
			next_node = neighber_list[next_id]
			G1.add_edge(now_node,next_node)
			count += 1
            
			now_node = next_node
			if nx.number_of_nodes(G1) >= n:
				break
        #print(now_node)
        #print(G.neighbors(now_node))
   
	return G1

def allRW(G,p):
	if p > 1:
		p = 1
	n = int(p*nx.number_of_nodes(G))
	G1 = nx.Graph()
	while nx.number_of_nodes(G1) < n:
		count = 0
		start_id = random.randint(0,nx.number_of_nodes(G)-1)
		s_node = nx.nodes(G)[start_id]
		now_node = s_node
		while count < 5*n:
            #print(now_node)
			neighber_list = G.neighbors(now_node)
		        for node in neighber_list:
                                G1.add_edge(node,now_node)
                        next_id = random.randint(0,len(neighber_list)-1)
			next_node = neighber_list[next_id]
		#	G1.add_edge(now_node,next_node)
			count += 1
            
			now_node = next_node
			if nx.number_of_nodes(G1) >= n:
				break
        #print(now_node)
        #print(G.neighbors(now_node))
   
	return G1

def CRW(G,p):
	if p > 1:
		p = 1
	n = int(p*nx.number_of_nodes(G))
       # print(n)
        G1 = nx.Graph()
	d = {}
	def Q(G,v):
		if d.has_key(v):
			return d[v]
		else:
			ans = nx.clustering(G,v)
			d[v] = ans
		return ans

    # a non-zero degree seed is selected at random
	while nx.number_of_nodes(G1) < n:
	        count = 0
	        tmp = 0
                while tmp <= 0:
                        start_id = random.randint(0,nx.number_of_nodes(G)-1)
                        s_node = nx.nodes(G)[start_id]
                        tmp = len(G.neighbors(s_node))

                now_node = s_node
                while count < 5*n:

                        neighber_list = G.neighbors(now_node)
                        next_id = random.randint(0,len(neighber_list)-1)
                        next_node = neighber_list[next_id]
                        p = random.random()
                        value = min(Q(G,next_node)+0.2,1.0)

                        if p <= value:

                                G1.add_edge(now_node,next_node)
                                now_node = next_node
                                count += 1
                        if(nx.number_of_nodes(G1) >= n):
                                return G1
	return G1
#提案手法　引数G 元のグラフ p:サンプリングする割合0<p<1
def BAS(G,p):
    N = 3
    G1 = nx.Graph() #サンプル後のグラフ
    e_list = []
    check2 = False
    n = int(p*nx.number_of_nodes(G))
    
    if N == 2:
        while True:
            start_id = random.randint(0,nx.number_of_nodes(G)-1)
            s_node = nx.nodes(G)[start_id]
            if len(G.neighbors(s_node))>0:
                neighber_list = G.neighbors(s_node)
                second_id = random.randint(0,len(neighber_list)-1)
                second_node = neighber_list[second_id]
                G1.add_edge(s_node,second_node)
                break
    elif N == 3:
        check = False
        while True:
            start_id = random.randint(0,nx.number_of_nodes(G)-1)
            s_node = nx.nodes(G)[start_id]
            if len(G.neighbors(s_node)) >= 2:
                check2 = False
                neighber_list = G.neighbors(s_node)
                for second_id in range(0,len(neighber_list)-1):
                    second_node = neighber_list[second_id]
                    for third_id in range(second_id+1,len(neighber_list)):
                        third_node = neighber_list[third_id]
                        t0 = (second_node,third_node)
                        t1 = (third_node,second_node)
                        if t0 in G.edges() or t1 in G.edges():
                            check = True
                            check2 = True
                            G1.add_edge(s_node,second_node)
                            G1.add_edge(second_node,third_node)
                            G1.add_edge(third_node,s_node)
                            break
                            
                    if check2:
                        break
            if check2:
                break
    else:
        N = 1
        start_id = random.randint(0,nx.number_of_nodes(G)-1)
        s_node = nx.nodes(G)[start_id]
        G1.add_node(s_node)
#ここまでがクリークの発見
    print("clique is OK")
    process = Blist() #サンプルされる候補のリスト(重複を含む)

    #まず最初のクリークの隣接ノードを候補に加える
    for node in G1.nodes():
        for neighbor in G.neighbors(node):
            if neighbor not in G1.nodes():
                process.insert(0,process.size-1,int(neighbor))
  #サンプルノード数がnに達するまでサンプリングを繰り返す
    while nx.number_of_nodes(G1) < n:
        next_node = str(process.select()) #候補からノードを選ぶ 

        stack = []
        
        G1nodes = G1.nodes()
        #選ばれたノードとサンプル済のノードをつなぐ
        for node in G.neighbors(next_node):
            if node in G1nodes:
                G1.add_edge(node,next_node)
        count = 3

###

#        while len(stack) > 0:

 #           num2 = random.randint(0,len(stack)-1)
  #          node = stack.pop(num2)
   #         G1.add_edge(node,next_node)
    #        count -= 0
     #       if count == 0:
      #          break


       ### 
#サンプルされたノードを候補リストから外す
        process.remove(0,process.size-1,next_node)
#候補リストを更新する
        G1nodes = G1.nodes()
        for node in G.neighbors(next_node):
            if node not in G1nodes:
                process.insert(0,process.size-1,int(node))
                
    return G1

def RWall(G,p):
    
    check = False
    
    while not check:
        count = 0
        tricount = 0
        tridic = {}
        degreedic = {}
        G1 = nx.Graph()
        n = int(p*nx.number_of_nodes(G))
        start_id = random.randint(0,nx.number_of_nodes(G)-1)
        s_node = nx.nodes(G)[start_id]
        G1.add_node(s_node)
    
        now_node = s_node
        tridic[now_node] = 0
        degreedic[now_node] = 0
        while True:
            neighbor_list = G.neighbors(now_node)
            next_id = random.randint(0,len(neighbor_list)-1)
            next_node = neighbor_list[next_id]
            trinodelist = []
            is_new_node = next_node not in G1.nodes()
            if is_new_node:
                G1.add_node(next_node)
                tridic[next_node] = 0
                degreedic[next_node] = 0

            count += 1
            if is_new_node:
                for node in G.neighbors(next_node):
                    if node in G1.nodes():
                        G1.add_edge(next_node,node)
                        if(degreedic.has_key(next_node)):
                            degreedic[next_node] = degreedic[next_node] + 1
                        else: 
                            degreedic[next_node] = 1
                        degreedic[node] = degreedic[node] + 1
                        trinodelist.append(node)
            
            if len(trinodelist) >= 2 and is_new_node:
                for i in range(0,len(trinodelist)-1):
                    for j in range(i+1,len(trinodelist)):
                        if trinodelist[j] in G1.neighbors(trinodelist[i]):
                            tricount += 1
                            if tridic.has_key(next_node):
                                tridic[next_node] = tridic[next_node] + 1
                            else:
                                tridic[next_node] = 1
                            if tridic.has_key(trinodelist[j]):
                                tridic[trinodelist[j]] = tridic[trinodelist[j]] + 1
                            else:
                                tridic[trinodelist[j]] = 1
                            if tridic.has_key(trinodelist[i]):
                                tridic[trinodelist[i]] = tridic[trinodelist[i]] + 1
                            else:
                                tridic[trinodelist[i]] = 1

            
            if random.random() < 0.75:
                next_id = random.randint(0,nx.number_of_nodes(G1)-1)
                next_node = G1.nodes()[next_id]
                now_node = next_node
            if is_new_node:
                now_node = next_node
                if nx.number_of_nodes(G1) >= n:
                    check = True
                    break
            if count > 50*n:
                break
    print("triangleの数"+str(tricount))
    # 
    ''''
    tri = list(nx.triangles(G1).values())
    sum1 = 0
    for num in tri:
        sum1 += int(num)
    '''
    print(graph.myGCC(G1,tricount))
    print(graph.GCC(G1))
    sumcluster = 0
    for node in G1.nodes():
        if degreedic.has_key(node):
            degree = degreedic[node]
            if degree >= 2 and tridic.has_key(node):
                sumcluster += 2.0*tridic[node]/(degree*(degree-1))

    print ("AverageCC:"+str(sumcluster/nx.number_of_nodes(G1)))

    return G1

def RWall2(G,p):
    check = False
    while not check:
        count = 0
        G1 = nx.Graph()
        n = int(p*nx.number_of_nodes(G))
        start_id = random.randint(0,nx.number_of_nodes(G)-1)
        s_node = nx.nodes(G)[start_id]
        G1.add_node(s_node)
    
        now_node = s_node
        while True:
            neighbor_list = G.neighbors(now_node)
            next_id = random.randint(0,len(neighbor_list)-1)
            next_node = neighbor_list[next_id]

            G1.add_node(next_node)
            count += 1
        
            if random.random() < 0:
                next_id = random.randint(0,nx.number_of_nodes(G1)-1)
                next_node = G1.nodes()[next_id]
            now_node = next_node
            if nx.number_of_nodes(G1) >= n:
                check = True
                break
            if count > 50*n:
                break
    
    return NodeConnect(G,G1)

def NodeConnect(G,G1):
    for node in G1.nodes():
        for node2 in G.neighbors(node):
            if node2 in G1.nodes():
                G1.add_edge(node,node2)
    return G1

