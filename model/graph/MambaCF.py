import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, bpr_k_loss, ccl_loss, directau_loss,  simce_loss, ssm_loss
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from util.args import get_params
import time
from scipy.sparse.linalg import svds
from model.graph.walks import get_random_walks
from mamba_ssm import Mamba
from torch_scatter import scatter_mean, scatter_sum

args = get_params()

class MambaCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MambaCF, self).__init__(conf, training_set, test_set)
        _args = OptionConf(self.config['MambaCF'])
        # self.n_layers = int(_args['-n_layer'])
        self.model = Mamba_encoder(self.data, self.emb_size, self.m_layers, self.bidirection, self.pos_enc, self.gcn)
        if args.loss in ['ssm', 'simce']:
            self.maxEpoch = 50

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            t0 = time.time()

            walk_node, _ = get_random_walks(self.model.adj, self.walk_length, sample_rate=self.sample_rate)
            walk_node = torch.from_numpy(walk_node).cuda()
        
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                
                rec_user_emb, rec_item_emb = model(walk_node)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                reg_loss = l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size


                if args.loss == 'bpr':
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'ssm':
                    rec_loss = ssm_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'simce':
                    rec_loss = simce_loss(user_emb, pos_item_emb, neg_item_emb, margin=args.margin)
                    
                    
                
                batch_loss = rec_loss + reg_loss 
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # if n % 100==0 and n>0:
                #     print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model(walk_node)
            if epoch % 1 == 0:
                self.fast_evaluation(epoch)
            print('each epoch: {} seconds'.format(time.time() - t0))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb



    def save(self):
        with torch.no_grad():

            # here is the evaluate
            walk_node, _ = get_random_walks(self.model.adj, self.test_walk_length, sample_rate=self.test_sample_rate)
            walk_node = torch.from_numpy(walk_node).cuda()


            
            self.best_user_emb, self.best_item_emb = self.model.forward(walk_node)

    def predict(self, u):
        u = self.data.get_user_id(u)
        user_emb = self.user_emb[u]
        item_emb = self.item_emb
        if args.loss in ['directau']:
            user_emb = F.normalize(user_emb, dim=-1)
            item_emb = F.normalize(item_emb, dim=-1)
            
        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score.cpu().numpy()


class Mamba_encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, bidirection=False, pos_enc=False, gcn=False, d_state=32, d_conv=8, expand=1, pos_emb=None):
        super(Mamba_encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj    
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.inter_matrix = data.ui_rat

        self.adj = data.ui_adj
        self.bidirect = bidirection
        self.pos_enc = pos_enc
        self.gcn=gcn

        if self.pos_enc:
            self.pos_emb = self.get_position_emb()

        

        self.seq_layers = torch.nn.ModuleList()
        if self.bidirect:
            self.seq_backward_layers = torch.nn.ModuleList()

        for _ in range(n_layers):
            self.seq_layers.append(Mamba(
                d_model=emb_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=None,
            ))
            if self.bidirect:
                self.seq_backward_layers.append(Mamba(
                    d_model=emb_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=None,
            ))
        

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def get_position_emb(self):
        
        inter_M = self.inter_matrix
        u, s, v = svds(inter_M, k=self.latent_size)
        user_pos = np.dot(u, np.diag(np.sqrt(s)))
        item_pos = np.dot(np.diag(np.sqrt(s)),v)
        item_pos = np.transpose(item_pos)

        user_pos = torch.from_numpy(user_pos)
        item_pos = torch.from_numpy(item_pos)
        pos_emb = torch.cat([user_pos, item_pos], dim=0)

        return pos_emb.cuda()

    def forward(self, walk_node):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        if self.pos_enc:
            ego_embeddings = ego_embeddings + self.pos_emb

        nnode = ego_embeddings.shape[0]
        
        all_embeddings = [ego_embeddings]

        
        for i in range(self.layers):
            x = ego_embeddings[walk_node]
            x_forward = self.seq_layers[i](x)
            if self.bidirect:
                x_backward = self.seq_backward_layers[i](x.flip([1]))
                x_backward = x_backward.flip([1])
                x_forward = (x_forward + x_backward) * 0.5
                del x_backward
            x = x_forward
            ego_embeddings = scatter_mean(x.reshape(-1, self.latent_size),
                         walk_node.flatten(),
                         dim=0,
                         dim_size=nnode)
            
        if self.gcn:
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            
            
        # all_embeddings = torch.stack(all_embeddings, dim=1)
        # all_embeddings = torch.mean(all_embeddings, dim=1)
        all_embeddings = ego_embeddings
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


