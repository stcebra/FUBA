import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from .utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):


        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)

        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = 100
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update
   
    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model
