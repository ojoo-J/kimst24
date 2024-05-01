import numpy as np
from tqdm.auto import tqdm

import torch

def hamming_distance(a,b):
    return torch.sum(a!=b)

def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2))

def find_k_nearest(query_activation, candi_activations, k):
    ham_list = []
    for i in tqdm(range(candi_activations.shape[0])):
        ham_list.append(hamming_distance(query_activation, candi_activations[i]).item())
    sorted_index = np.argsort(ham_list)
    return np.sort(ham_list), sorted_index[1:k+1]

def get_local_nearest_idx(query_activation, candi_activations, th=30000):
    dist_list = []
    for i in range(candi_activations.shape[0]):
        dist_list.append(hamming_distance(query_activation, candi_activations[i]).item())
    local_nearest_idx = np.where(np.array(dist_list)<th)[0].tolist()
    return local_nearest_idx # hamming distance 기준 th 이내에 존재하는 샘플들의 인덱스

def get_local_nearest_idx_euc(query_hidden, candi_hiddens, th=200):
    dist_list = []
    for i in range(candi_hiddens.shape[0]):
        dist_list.append(euclidean_distance(query_hidden, candi_hiddens[i]).item())
    local_nearest_idx = np.where(np.array(dist_list)<th)[0].tolist()
    return local_nearest_idx # euclidean distance 기준 th 이내에 존재하는 샘플들의 인덱스

def get_local_nearest_ham_and_idx(query_activation, candi_activations, th=70000):
    local_nearest_dist = []
    local_nearest_idx = []
    for i, idx in enumerate(range(candi_activations.shape[0])):
        h = hamming_distance(query_activation, candi_activations[idx]).item()
        if h <= th:
            local_nearest_dist.append(h)
            local_nearest_idx.append(i)
    return local_nearest_idx, local_nearest_dist # hamming distance 기준 th 이내에 존재하는 샘플들의 인덱스, hamming distance

def get_local_nearest_euc_and_idx(query_hidden, candi_hiddens, th=200):
    local_nearest_dist = []
    local_nearest_idx = []
    for i, idx in enumerate(range(candi_hiddens.shape[0])):
        h = euclidean_distance(query_hidden, candi_hiddens[idx]).item()
        if h <= th:
            local_nearest_dist.append(h)
            local_nearest_idx.append(i)
    return local_nearest_idx, local_nearest_dist # hamming distance 기준 th 이내에 존재하는 샘플들의 인덱스, hamming distance