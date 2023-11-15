import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from torch_geometric.data import Dataset

def prepare_graph(data, dataset):
    # Populating edge lists in Graph object based on edge_list
    print("Populating edge lists into Graph object")
    edge_index_dict = data.edge_index_dict 
    graph = Graph()
    edg = graph.edge_list
    years = data.node_year['paper'].t().numpy()[0]
    # for every type of edge relation i.e. ('author', 'affiliated_with', 'institution'), ...
    for key in edge_index_dict:
        print(key) # print relation name
        edges = edge_index_dict[key] 
        '''
        tensor( [[      0,       1,       2,  ..., 1134645, 1134647, 1134648],
                [    845,     996,    3197,  ...,    5189,    4668,    4668]]) example edges tensor
        '''
        # getting types of source, relation and edge ('author', 'affiliated_with', 'institution')
        s_type, r_type, t_type = key[0], key[1], key[2]
        elist = edg[t_type][s_type][r_type]
        rlist = edg[s_type][t_type]['rev_' + r_type]
        # adding year if the type is paper
        for s_id, t_id in edges.t().tolist():
            year = None
            if s_type == 'paper':
                year = years[s_id]
            elif t_type == 'paper':
                year = years[t_id]
            elist[t_id][s_id] = year
            rlist[s_id][t_id] = year

    print("")
    # Reformatting edge list and computing node degrees
    print("Reformatting edge lists and computing node degrees")
    edg = {}
    deg = {key : np.zeros(data.num_nodes_dict[key]) for key in data.num_nodes_dict}
    for k1 in graph.edge_list:
        if k1 not in edg:
            edg[k1] = {}
        for k2 in graph.edge_list[k1]:
            if k2 not in edg[k1]:
                edg[k1][k2] = {}
            for k3 in graph.edge_list[k1][k2]:
                if k3 not in edg[k1][k2]:
                    edg[k1][k2][k3] = {}
                for e1 in graph.edge_list[k1][k2][k3]:
                    if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                        continue

                    edg[k1][k2][k3][e1] = {}
                    for e2 in graph.edge_list[k1][k2][k3][e1]:
                        edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
                    deg[k1][e1] += len(edg[k1][k2][k3][e1])
                print(k1, k2, k3, len(edg[k1][k2][k3]))
    graph.edge_list = edg # inserting new edge list into Graph object

    print("")
    # Constructing node feature vectors for each node type in graph
    print("Constructing node feature vectors for each node type in graph")
    paper_node_features = data.x_dict['paper'].numpy() # data into numpy
    # append log degree to get full paper node features
    graph.node_feature['paper'] = np.concatenate((paper_node_features, np.log10(deg['paper'].reshape(-1, 1))), axis=-1)
    # These are node types: {'author': 1134649, 'field_of_study': 59965, 'institution': 8740, 'paper': 736389}
    for node_type in data.num_nodes_dict:
        print(node_type)
        if node_type not in ['paper', 'institution']:
            i = []
            for rel_type in graph.edge_list[node_type]['paper']:
                for t in graph.edge_list[node_type]['paper'][rel_type]:
                    for s in graph.edge_list[node_type]['paper'][rel_type][t]:
                        i += [[t,s]]
                if len(i) == 0:
                    continue
            i = np.array(i).T
            v = np.ones(i.shape[1])
            m = normalize(sp.coo_matrix((v, i), \
                shape=(data.num_nodes_dict[node_type], data.num_nodes_dict['paper'])))
            out = m.dot(paper_node_features)
            graph.node_feature[node_type] = np.concatenate((out, np.log10(deg[node_type].reshape(-1, 1))), axis=-1)

    print("")
    # Contructing node feature vectors for institution nodes
    print("Constructing Node features for institutions")    
    cv = graph.node_feature['author'][:, :-1]
    i = []
    for _rel in graph.edge_list['institution']['author']:
        for j in graph.edge_list['institution']['author'][_rel]:
            for t in graph.edge_list['institution']['author'][_rel][j]:
                i += [[j, t]]
    i = np.array(i).T
    v = np.ones(i.shape[1])
    m = normalize(sp.coo_matrix((v, i), \
        shape=(data.num_nodes_dict['institution'], data.num_nodes_dict['author'])))
    out = m.dot(cv)
    
    graph.node_feature['institution'] = np.concatenate((out, np.log10(deg['institution'].reshape(-1, 1))), axis=-1)      

    # y_dict
    y = data.y_dict['paper'].t().numpy()[0]

    print("")
    # Splitting dataset into training, validation and testing
    print("Splitting dataset into train, val and test")
    split_idx = dataset.get_idx_split()
    train_paper = split_idx['train']['paper'].numpy()
    valid_paper = split_idx['valid']['paper'].numpy()
    test_paper  = split_idx['test']['paper'].numpy()

    graph.y = y
    graph.train_paper = train_paper
    graph.valid_paper = valid_paper
    graph.test_paper  = test_paper
    graph.years       = years

    print("")
    print("Creating Masks")
    graph.train_mask = np.zeros(len(graph.node_feature['paper']), dtype=bool)
    graph.train_mask[graph.train_paper] = True

    graph.valid_mask = np.zeros(len(graph.node_feature['paper']), dtype=bool)
    graph.valid_mask[graph.valid_paper] = True

    graph.test_mask = np.zeros(len(graph.node_feature['paper']),  dtype=bool)
    graph.test_mask[graph.test_paper] = True

    print("")
    # Preprocessing graph object is now complete
    print("Preprocessing complete")

    return graph, y, train_paper, valid_paper, test_paper

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        times[_type]   = tims
        indxs[_type]   = idxs
        
        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts

def feature_MAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()), dtype = np.int)
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        feature[_type] = graph.node_feature[_type][idxs]
        times[_type]   = tims
        indxs[_type]   = idxs
        
    return feature, times, indxs, texts

def sample_subgraph(graph, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_OAG):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id: [ser, time]}
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))
    new_layer_adj  = defaultdict( #target_type
                                    lambda: defaultdict(  #source_type
                                        lambda: defaultdict(  #relation_type
                                            lambda: [] #[target_id, source_id]
                                )))
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    if source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time

 
    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
    '''
    feature, times, indxs, texts = feature_extractor(layer_data, graph)
            
    edge_list = defaultdict( #target_type
                        lambda: defaultdict(  #source_type
                            lambda: defaultdict(  #relation_type
                                lambda: [] # [target_id, source_id] 
                                    )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
    '''
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld  = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    return feature, times, edge_list, indxs, texts

def to_torch(feature, time, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_time    = []
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num     += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_time    += list(time[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]
        
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type  += [edge_dict[relation_type]]   
                    '''
                        Our time ranges from 1900 - 2020, largest span is 120.
                    '''
                    edge_time  += [node_time[tid] - node_time[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_time    = torch.LongTensor(edge_time)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict

'''
Graph object implementation from original HGT paper
This is thier implementation, all credit goes to the authors
'''

from collections import defaultdict

class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and backward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())

def randint():
    return np.random.randint(2**31 - 1)
   
def ogbn_sample(seed, samp_nodes, graph, sample_depth, sample_width):
    np.random.seed(seed)
    ylabel      = torch.LongTensor(graph.y[samp_nodes])
    #graph, time_range, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_MAG
    feature, times, edge_list, indxs, _ = sample_subgraph(
        graph,
        inp = {'paper': np.concatenate([samp_nodes, graph.years[samp_nodes]]).reshape(2, -1).transpose()},
        sampled_depth = sample_depth, 
        sampled_number = sample_width,
        feature_extractor = feature_MAG)
    
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(feature, times, edge_list, graph)
    
    train_mask = graph.train_mask[indxs['paper']]
    valid_mask = graph.valid_mask[indxs['paper']]
    test_mask  = graph.test_mask[indxs['paper']]
    ylabel     = graph.y[indxs['paper']]
    return node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel

def prepare_data_train(pool, n_batch, batch_size, target_nodes, graph, sample_depth, sample_width):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []

    for batch_id in np.arange(n_batch):
        print(f'starting preprocessing batch: {batch_id}')
        p = pool.apply_async(ogbn_sample, args=([
            randint(), \
            np.random.choice(target_nodes, batch_size, replace = False),
            graph,
            sample_depth,
            sample_width]))
        jobs.append(p)
        print(f'finished preprocessing batch: {batch_id}')
    print("Preprocessing complete ---------------------------------------------------")
    return jobs