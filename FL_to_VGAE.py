from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import math, time
import torch
import model
import torch.nn.functional as F
import networkx as nx
from torch.optim import Adam
from scipy.sparse import csr_matrix
from collections import defaultdict
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges

def fl_to_vgae(parameters,  FL_adj):
    A_new = nx.from_numpy_matrix(FL_adj)
    graph_data = list(A_new.edges())
    number_of_train = math.ceil(3 * np.shape(FL_adj)[1] / 3)
    total_clients, num_features = parameters.shape

    post_parameters = np.where(parameters < 0.5, 0, 1)
    train_parameters = post_parameters[0:number_of_train, :]
    test_parameters = post_parameters[number_of_train: total_clients, :]
    allx = csr_matrix(train_parameters)
    tx = csr_matrix(test_parameters)
    test_idx_reorder = list(range(number_of_train, total_clients))
    test_idx_range = np.sort(test_idx_reorder)
    graph = defaultdict(list)

    for (key, value) in graph_data:
        graph[key].append(value)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)  ### bug
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    #no norm process
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                        torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer
    vgae = getattr(model,'VGAE')(adj_norm)
    optimizer = Adam(vgae.parameters(), lr=0.01)


    # train model
    for epoch in range(num_features):
        A_pred = vgae(features) # adjacency matrix of output
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                        weight=weight_tensor)
        kl_divergence = 0.5 / A_pred.size(0) * (
                        1 + 2 * vgae.logstd - vgae.mean ** 2 - torch.exp(vgae.logstd) ** 2).sum(1).mean()
        loss -= kl_divergence
        #
        loss.backward()
        optimizer.step()

    L_G = nx.from_numpy_matrix(FL_adj)
    L = nx.laplacian_matrix(L_G)
    u, s, vh = np.linalg.svd(L.toarray(), full_matrices=True)

    z_numpy = A_pred.detach().numpy()
    z_G = nx.from_numpy_matrix(z_numpy)
    # Li_new = np.diag(z_numpy.diagonal()) - z_numpy
    Li = nx.laplacian_matrix(z_G)
    u_new, s_new, vh_new = np.linalg.svd(Li.toarray(), full_matrices=True)

    # np.random.seed(42)
    new_features = np.dot(u_new, np.dot(u.T, parameters))
    """mnist dataset"""
    w_attack = np.sum(new_features, axis=1) / new_features.shape[1] * np.random.uniform(-3.0, 0.1, new_features.shape[0])  # new_features.shape[0] is column
    """cifar10 dataset"""
    # w_attack = np.sum(new_features, axis=0) / new_features.shape[0] * np.random.normal(0.0, 1.0, new_features.shape[1]) # new_features.shape[0] is column

    # w_attack = np.sum(new_features, axis=0) / 100 # new_features.shape[0] is column
    dist = abs(np.linalg.norm(w_attack - new_features[:,[0]]) - np.linalg.norm(w_attack - new_features[:,[1]]))
    return w_attack, z_numpy, dist

