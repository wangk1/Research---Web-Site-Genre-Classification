__author__ = 'Kevin'

import collections as coll, itertools as iter,sys,random as rand
import igraph
import numpy as np,scipy.sparse as sp
import math

from sklearn.neighbors import NearestNeighbors

from util.base_util import normalize_genre_string

"""
Graph cut algorithm with Markov random field

Graph Representation:
    The graph is represented with igraph's Graph package

    A directed graph with copies of edges is used to mimic MRF

    Vetex has the following property
        label-The label assigned to the vertex, note may be different based on what needs to visualized
        pred_label-The prediction label, does not exist in label vertex
        true_label-The actual label
        color-Color of the label
        is_label-True/False if the vertex is a label vertex
        ref_id=Reference id of the vertex, used for back checking
        index=the index in the graph cut params input matrix

    Edge has the following property
        is_terminate-True/False if this edge is connecting a vertex to a terminal vertex, aka label
        weight-The weight of the edge

"""
class GraphCutParams:
    """
    The class used for holding the parameters to the entire graph cut algorithm
    """
    def __init__(self,*,X,y,ref_id,k_closest_neighbors,genre_level=1,vocab_size,num_clusters):
        self.X=X
        self.y=y
        self.ref_id=ref_id
        self.k_closest_neighbors=k_closest_neighbors
        self.e_data_weight=1
        self.e_smooth_weight=0.2

        self.e_data=None
        self.e_smooth=None

        self.genre_level=genre_level
        self.vocab_size=vocab_size

        self.num_cluster=num_clusters

        #auto genereate colors of vertex and the vertex labels
        self.cluster_names=["c{}".format(i) for i in range(0,self.num_cluster)]
        self.color_map={cn:(rand.randint(0,255),rand.randint(0,255),rand.randint(0,255)) for cn in self.cluster_names}

        """
        Used to find the P(a,b)/P(b) where P(b) is the probability that are certain class will occur.
            P(a,b) is the joint
        """
        self.genre_to_index={}
        self.genre_word_count=None #stores the genre to word count vector mapping
        self.label_occurence_map=None #maps label string to the number of occurence

        self.label_to_neighbor_vertex=coll.defaultdict(lambda: []) #map from a vertex to its x neighbors


def alpha_beta_swap(graph_cut_data):
    """


    :param graph_cut_data:
    :return:
    """
    assert isinstance(graph_cut_data,GraphCutParams)

    #change labels to stirng for name purpose
    graph_cut_data.ref_id=np.array([str(i) for i in graph_cut_data.ref_id])
    print("Creating MRF")
    #add vertices, assign randomly website vetex to a label vertex
    incomplete_mrf=create_mrf(graph_cut_data)

    print("Randomly Assigning Labels")
    incomplete_mrf=random_assign_labels(incomplete_mrf,graph_cut_data)


    mrf=attach_edge(incomplete_mrf,graph_cut_data)

    #Grab the distribution of word with genre. Occurence of each genre
    graph_cut_data.label_occurence_map,graph_cut_data.genre_word_count=recalibrate_distribution(mrf,graph_cut_data)
    #assign the weights to the edges
    recalibrate_label_edge_weights(mrf,graph_cut_data,graph_cut_data.label_occurence_map,graph_cut_data.genre_word_count)

    success=True
    prev_energy=calculate_energy(mrf)
    while success:
        #label set of vertex object tuples
        label_set=iter.combinations(graph_cut_data.label_vector_list,2)

        success=False
        for iteration,(alpha,beta) in enumerate(label_set):
            

            #graph cut from source node to target node, with weight as the attribute used
            #as the algorithm
            min_cut=mrf.st_mincut(alpha,beta,"weight")

            #save the graph_cut_data's data before we calculate partition


            #partition returns the old mrf if the new energy is greater than prev energy
            new_mrf,label_occurence_map,genre_word_count=partition(mrf,min_cut,alpha["name"],beta["name"],graph_cut_data)
            new_energy=calculate_energy(new_mrf)
            
            print("Iteration {}, new energy: {} old energy: {}".format(iteration,new_energy,prev_energy))

            if new_energy<prev_energy:
                print("Found better energy")
                success=True

                graph_cut_data.genre_word_count=genre_word_count
                graph_cut_data.label_occurence_map=label_occurence_map

                mrf=new_mrf

                prev_energy=new_energy

    #assign colors and labels
    for vertex in mrf.vs:
        vertex["label"]=vertex["pred_label"]
        vertex["color"]=graph_cut_data.color_map[vertex["pred_label"]]

    layout=mrf.layout_kamada_kawai()
    igraph.plot(mrf,layout=layout)
    print("Done")


def alpha_expansion(graph_cut_data):
    assert isinstance(graph_cut_data,GraphCutParams)
    pass

def create_mrf(graph_cut_data):
    """
    Create the mrf without any edges but containing all the vertices.

    The vertices are the individual website vertex and the label vertices.

    Also, initialize the coo_matrix representation of genre word distribution

    :param graph_cut_data:
    :return mrf: The graph of vertices, label_vertices: vertex seq object of all vertexes that are labels
    """

    assert isinstance(graph_cut_data,GraphCutParams)
    mrf=igraph.Graph(directed=True)

    label_vertices=[]

    if not isinstance(graph_cut_data.y.shape[0],str):
        actual_labels=[[normalize_genre_string(g,1) for g in g_list] for g_list in graph_cut_data.y]
    else:
        actual_labels=[i for i in graph_cut_data.y]

    #create the label to vocab matrix
    graph_cut_data.genre_word_count=sp.coo_matrix((graph_cut_data.num_cluster,graph_cut_data.vocab_size),dtype=np.dtype(float))

    #create the label vertex
    for index,l in enumerate(graph_cut_data.cluster_names):
        mrf.add_vertex(l,pred_label=l,actual_label=l,is_label=True,index=-1)
        label_vertices.append(l)
        graph_cut_data.genre_to_index[l]=index

    graph_cut_data.label_vector_list=[mrf.vs.find(name=label_vertex) for label_vertex in label_vertices]

    #create the website vertices
    for index,ref_id in enumerate(graph_cut_data.ref_id):
        mrf.add_vertex(ref_id,is_label=False,index=index,actual_label=actual_labels[index])

    print("Actual Labels length: {}".format(len(actual_labels)))

    return mrf

def random_assign_labels(mrf,graph_cut_data):
    """
    Assign edges to each vertex in mrf connecting it to a random label

    :param mrf:
    :returns assignment: assignment of label to different
    """
    assert isinstance(mrf,igraph.Graph)

    #assign labels randomly
    for vertex in mrf.vs(is_label=False):

        lbl_assignment=graph_cut_data.label_vector_list[rand.randint(0,len(graph_cut_data.label_vector_list)-1)]

        mrf.vs.find(vertex["name"])["pred_label"]=lbl_assignment["name"]

    return mrf

def calculate_energy(mrf):
    """
    Calculate the energy of an mrf by adding up the weights of all the edges

    E(f)= E_smooth(f)+E_data(f)
    :param mrf:
    :return: energy, float
    """
    return sum(mrf.es["weight"])/2

def partition(mrf,cut_list,l1,l2,graph_cut_data):
    """
    Partition the mrf based on the list of edges cut with the st_mincut.

    We only reassign genres from label1 and label2

    Calculate the new distribution

    :param mrf:
    :param cut_list:
    :return:
    """

    new_mrf=mrf.copy()

    #reassign the classes
    for min_cut in cut_list.es:
        source=mrf.vs.find(min_cut.source)
        target=mrf.vs.find(min_cut.target)

        if not source["is_label"] and not target["is_label"]:
            continue

        label_vertex,non_lbl_vertex=(target,source) if not source["is_label"] \
            else (source,target)

        #only change the right vertexes
        if non_lbl_vertex["pred_label"] not in {l1,l2}:
            continue

        #reassign class
        non_lbl_vertex["pred_label"]=label_vertex["name"]

    #recalibrate the class distribution, we only label the classes with label alpha and beta
    label_occurence_map,genre_word_count=recalibrate_distribution(new_mrf,graph_cut_data)
    recalibrate_label_edge_weights(new_mrf,graph_cut_data,label_occurence_map,genre_word_count,only_relabel={l1,l2})

    return new_mrf,label_occurence_map,genre_word_count

def attach_edge(mrf,graph_cut_data):
    X=graph_cut_data.X
    y=graph_cut_data.y

    #first attach, the non terminal V to the other non teminal V's with nearest neighbor
    nn=NearestNeighbors(n_neighbors=graph_cut_data.k_closest_neighbors)
    nn.fit(X,y)

    dist,ind=nn.kneighbors(X,n_neighbors=graph_cut_data.k_closest_neighbors+1)

    for index,(k_closest,k_dist) in enumerate(zip(ind,dist)):
        curr_name=graph_cut_data.ref_id[index]

        name_to_weight={}
        for neighbor_index,neighbor_dist in zip(k_closest,k_dist):

            #skip because the current one will register as a neighbor
            if neighbor_index == index:
                continue
            neighbor_name=graph_cut_data.ref_id[neighbor_index]

            name_to_weight[neighbor_name]=neighbor_dist

            graph_cut_data.label_to_neighbor_vertex[curr_name].append(mrf.vs.find(neighbor_name))

        total_dist=sum([v for k,v in name_to_weight.items()])
        for neighbor_name,neighbor_dist in name_to_weight.items():
            mrf.add_edge(curr_name,neighbor_name,is_terminate=False,weight=neighbor_dist/total_dist*graph_cut_data.e_data_weight)
            mrf.add_edge(neighbor_name,curr_name,is_terminate=False,weight=neighbor_dist/total_dist*graph_cut_data.e_data_weight)


    #attach the terminals
    for not_label_vs in mrf.vs(is_label=False):
        for label_vertex in graph_cut_data.label_vector_list:
            mrf.add_edge(not_label_vs,label_vertex,weight=0,is_terminate=True)
            mrf.add_edge(label_vertex,not_label_vs,weight=0,is_terminate=True)

    return mrf

def recalibrate_distribution(mrf,graph_cut_data):
    genre_word_count=[0]*len(graph_cut_data.label_vector_list)
    label_occurence_map={}

    label_to_index={label_vertex["name"]:c for c,label_vertex in enumerate(graph_cut_data.label_vector_list)}

    #the webpage to the index in the label list of its label
    labels_index=np.array([label_to_index[not_label_vertex["pred_label"]] for not_label_vertex in mrf.vs(is_label=False)])

    for index,label_vertex in enumerate(graph_cut_data.label_vector_list):
        selector=labels_index==index

        #group all X that have the same label with the corresponding index
        genre_word_count[index]=graph_cut_data.X[selector].sum(0)
        #also find how often the class occurs
        label_occurence_map[label_vertex["name"]]=np.sum(selector)

    return label_occurence_map,genre_word_count

def recalibrate_label_edge_weights(new_mrf,graph_cut_data,label_occurence_map,genre_word_count,only_relabel=None):
    genre_word_count=graph_cut_data.genre_word_count


    #update each terminal edge weight
    for terminal_edge in new_mrf.es(is_terminate=True):
        target=new_mrf.vs.find(terminal_edge.target)
        source=new_mrf.vs.find(terminal_edge.source)
        label_vertex,non_labl_vertex=(target, source) if target["is_label"] else \
                    (source, target)

        if only_relabel is not None and label_vertex["name"] not in only_relabel:
            continue

        #P(a|b)=P(a and b)/P(b)=(N(a and b)/(N(a and b) + N( not a and b))/(N(class)/N(allclass))
        p_a_b_single=(graph_cut_data.X[int(non_labl_vertex["index"])]/genre_word_count[graph_cut_data.genre_to_index[label_vertex["name"]]].sum())
        p_a_b=p_a_b_single.dot(p_a_b_single.T)[0,0]

        p_b=label_occurence_map[label_vertex["name"]]/graph_cut_data.X.shape[0]

        p_a_given_b=p_a_b/p_b

        data_edge_score=1 if p_a_given_b>1 else 1-p_a_given_b

        neighbor_labels=[v["pred_label"] for v in graph_cut_data.label_to_neighbor_vertex[non_labl_vertex["name"]]]
        neighbor_score=sum((i != target["name"] for i in neighbor_labels))/len(neighbor_labels)

        #1-P(a|b) to minimize better edges
        terminal_edge["weight"]=(data_edge_score+neighbor_score)/2*graph_cut_data.e_data_weight