import collections
import random
import time
import networkx as nx
import matplotlib.pyplot as plt

def load_graph(path):
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()
            v_i = int(vertices[0])
            v_j = int(vertices[1])
            w = 1.0 # If the dataset has weights, read the weights from the dataset
            G[v_i][v_j] = w
            G[v_j][v_i] = w
    return G


# Node class that stores community and node ID information
class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        # node id
        self._vid = vid
        # community id
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in  # weight of edges inside the node


class Louvain:
    def __init__(self, G):
        self._G = G
        self._m = 0  # number of edges  The graph will coagulate and change dynamically
        self._cid_vertices = {}  # (community number, set of node numbers included)
        self._vid_vertex = {}  # (node number, corresponding Vertex instance)
        self.C_list = []  # store the community structure of each coagulation
        for vid in self._G.keys():
            # at the beginning, each point is a community
            self._cid_vertices[vid] = {vid}
            # at the beginning, the community number is the node number
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})
            # calculate the number of edges, maintaining one edge between every two points
            self._m += sum([1 for neighbor in self._G[vid].keys()
                           if neighbor > vid])

    # modularity optimization phase
    def first_stage(self):
        mod_inc = False  # used to judge whether the algorithm can be terminated
        visit_sequence = self._G.keys()
        # random access
        random.shuffle(list(visit_sequence))
        while True:
            can_stop = True  # whether the first stage can be terminated
            # traverse all nodes
            for v_vid in visit_sequence:
                # get the community number of the node
                v_cid = self._vid_vertex[v_vid]._cid
                # k_v is the weight (degree) of the node internal and external edge weight sum
                k_v = sum(self._G[v_vid].values()) + \
                    self._vid_vertex[v_vid]._kin
                # initialize the community number of the node to be the community number of the node itself
                cid_Q = {}
                # traverse all neighbors of the node
                for w_vid in self._G[v_vid].keys():
                    # get the community number of the neighbor node
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        # tot is the total weight of the node in the community
                        # tot is the sum of the weights on the links associated with nodes in community C
                        tot = sum(
                            [sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        # k_v_in is the weight of the internal edge of the node in the community
                        k_v_in = sum(
                            [v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        # delta_Q is the change in modularity after the node is moved to the community
                        delta_Q = k_v_in - k_v * tot / self._m
                        cid_Q[w_cid] = delta_Q

                # find the community with the largest increase in modularity
                cid, max_delta_Q = sorted(
                    cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                if max_delta_Q > 0.0 and cid != v_cid:
                    # if the node is moved to the community, the community number of the node is updated
                    self._vid_vertex[v_vid]._cid = cid
                    # add this node to this community
                    self._cid_vertices[cid].add(v_vid)
                    # remove this node from the original community
                    self._cid_vertices[v_cid].remove(v_vid)
                    # update the community number of the neighbor node
                    # we keep this for the modularity maximization phase
                    can_stop = False
                    mod_inc = True
            if can_stop:
                break
            self.C_list.append(self._cid_vertices)
        return mod_inc

    # community aggregation phase
    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        # traverse all communities
        for cid, vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            # Iterate over all nodes in the community Find the total weight within the community Find the union of nodes within the community
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                # k,v  Calculate the total weight within the kin community for the 
                # neighbors and the weights of the edges between them. Here we iterate through every neighbor of vid 
                # within the community because the edges are shared by two points and will be calculated later /2
                for k, v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            # update the community number of the node
            cid_vertices[cid] = {cid}
            vid_vertex[cid] = new_vertex

        G = collections.defaultdict(dict)
        # traverse all communities
        for cid1, vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                # Find another community that is not empty after cid
                if cid2 <= cid1 or len(vertices2) == 0:
                    continue
                edge_weight = 0.0
                # Iterate over the points in the cid1 community
                for vid in vertices1:
                    # Iterate over the weights of the edges already between the neighbors
                    #  of this point in community 2 (i.e., the total weight of the edges 
                    # between the two communities Consider multiple edges as one edge)
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        # Update communities and points Each community is seen as a point
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G  



    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities

    def execute(self):
        iter_time = 1
        while True:
            iter_time += 1
            # 反复迭代，直到网络中任何节点的移动都不能再改善总的 modularity 值为止
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
            
        return self.get_communities()


# 可视化划分结果
def showCommunity(G, partition, pos):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    labels = {}
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(nodeID) + '$'  # 设置可视化label
            cluster[nodeID] = index  # 节点分区号

    # 可视化节点
    colors = ['r', 'g', 'b', 'y', 'm']
    shapes = ['v', 'D', 'o', '^', '<']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index],
                               node_shape=shapes[index],
                               node_size=350,
                               alpha=1)

    # 可视化边
    edges = {len(partition): []}
    for link in G.edges():
        # cluster间的link
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的link
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)

    for index, edgelist in enumerate(edges.values()):
        # cluster内
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index])
        else:
            # cluster间
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=3, alpha=0.8, edge_color=colors[index])

    # 可视化label
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.axis('off')
    plt.show()


def cal_Q(partition, G):  # 计算Q
    m = len(G.edges(None, False))
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  
        t = 0.0
        for node in community: 
            # G.neighbors(node)
            t += len([x for x in G.neighbors(node)])
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

class Graph:
    graph = nx.DiGraph()

    def __init__(self):
        self.graph = nx.DiGraph()

    def createGraph(self, filename):
        file = open(filename, 'r')

        for line in file.readlines():
            nodes = line.split()
            edge = (int(nodes[0]), int(nodes[1]))
            self.graph.add_edge(*edge)

        return self.graph

    # G = load_graph('data/club.txt')
print("hello")
G = load_graph('data/karate.txt')
obj = Graph()
G1 = obj.createGraph("Data//karate.txt")
# G1 = nx.karate_club_graph()
# pos = nx.spring_layout(G1)
start_time = time.time()
algorithm = Louvain(G)
communities = algorithm.execute()
end_time = time.time()

communities = sorted(communities, key=lambda b: -len(b))  
count = 0
for communitie in communities:
    count += 1
    print("community number ", count, " ", communitie)
print(cal_Q(communities, G1))
print(f'time total : {end_time - start_time}')

pos = nx.spring_layout(G1)
# save communities graph into txt file
# dot -Tpng graphviz.txt -o graphviz.png

showCommunity(G1, communities, pos)