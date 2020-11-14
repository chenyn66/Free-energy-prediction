
#%%
import pystan
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
from sklearn.metrics import *
import sys

class Node_model:

    # def __init__(self,prior=(0,3),good=(0,1),bad=(5,3),pbad=0.1): # TODO: N(0,1) assumption
    #     cycle_code = """
    #     data {{
    #         int<lower=1> N;    // Total number of nodes
    #         int<lower=1> M;    // Total number of edges
    #         int<lower=1> src[M]; // Edge connectivity
    #         int<lower=1> dst[M]; 
    #         real y[M];         // measured edge data
    #     }}
    #     parameters {{
    #         vector[N] v;       // value at each node
    #     }}
    #     model {{
    #         real pBad={pbad};
    #         v ~ normal({pu}, {ps});
    #         for (e in 1:M) {{
    #         target += log_sum_exp( log(pBad/2) + normal_lpdf( v[dst[e]] - v[src[e]] - y[e] | {nbu}, {bs}), // bad noise assumption 
    #                     log_sum_exp(
    #                                 log(1-pBad) + normal_lpdf( v[dst[e]] - v[src[e]] -y[e] | {gu}, {gs}), // good noise assumption 
    #                                 log(pBad/2) + normal_lpdf( v[dst[e]] - v[src[e]] -y[e] | {pbu}, {bs}) ) ); // bad noise assumption 
    #         }}
    #     }}
    #     """.format(pbad=pbad,pu=prior[0],ps=prior[1],gu=good[0],gs=good[1],nbu=-bad[0],pbu=bad[0],bs=bad[1])
    #     self.prior = prior
    #     self.good = good
    #     self.bad = bad
    #     self.pbad = pbad
    #     self.model = pystan.StanModel(model_code=cycle_code)


    def __init__(self,prior=(0,3),noise=(0,1)): # TODO: N(0,1) assumption

        if prior:
            cycle_code = """
            data {{
                int<lower=1> N;    // Total number of nodes
                int<lower=1> M;    // Total number of edges
                int<lower=1> src[M]; // Edge connectivity
                int<lower=1> dst[M]; 
                real y[M];         // measured edge data
            }}
            parameters {{
                vector[N] v;       // value at each node
            }}
            model {{
                v ~ normal({pu}, {ps});
                for (e in 1:M) {{
                target += normal_lpdf( v[dst[e]] - v[src[e]] - y[e] | {gu}, {gs}); 
                }}
            }}
            """.format(pu=prior[0],ps=prior[1],gu=noise[0],gs=noise[1])
        else:
            cycle_code = """
            data {{
                int<lower=1> N;    // Total number of nodes
                int<lower=1> M;    // Total number of edges
                int<lower=1> src[M]; // Edge connectivity
                int<lower=1> dst[M]; 
                real y[M];         // measured edge data
            }}
            parameters {{
                vector[N] v;       // value at each node
            }}
            model {{
                for (e in 1:M) {{
                target += normal_lpdf( v[dst[e]] - v[src[e]] - y[e] | {gu}, {gs}); 
                }}
            }}
            """.format(gu=noise[0],gs=noise[1])
        self.prior = prior
        self.model = pystan.StanModel(model_code=cycle_code)

    def read_graph(self, graph_dir):
        start = []
        end = []
        y = []
        graph = open(graph_dir)
        idx = 0
        self.node2idx = dict()
        self.idx2node = dict()

        for line in graph:
            s,e,v = line.rstrip().split(' ')
            if s not in self.node2idx:
                self.node2idx[s] = idx
                self.idx2node[idx] = s
                idx += 1

            if e not in self.node2idx:
                self.node2idx[e] = idx
                self.idx2node[idx] = e
                idx += 1
            
            start.append(self.node2idx[s]+1)
            end.append(self.node2idx[e]+1)
            y.append(float(v))

        self.cycle_dat = {
                    'N':len(self.node2idx),
                    'M':len(y),
                    'src':start,
                    'dst':end,
                    'y': y }


    def sample(self, it=1000,chains=4,verbose=True):
        fit = self.model.sampling(data=self.cycle_dat, iter=it, chains=chains,verbose=verbose)
        self.result = fit
        self.sample = True
        return fit

    def optimize(self):
        fit = self.model.optimizing(data=self.cycle_dat)
        self.result = fit
        self.sample = False
        return fit

    def pBgV(self, pred_diff,obs_diff):
        noise = obs_diff-pred_diff
        pB = 0.5*stats.norm.pdf(noise,*self.bad)+0.5*stats.norm.pdf(noise,-self.bad[0],self.bad[1])
        pB *= self.pbad
        pG = stats.norm.pdf(noise,*self.good)*(1-self.pbad)
        return pB/(pB+pG)

    def predict(self):
        samples = self.result.extract()['v']
        start = self.cycle_dat['src']
        end = self.cycle_dat['dst']
        y = self.cycle_dat['y']

        edge_pred = dict()
        for i in range(len(start)):
            s = start[i]-1
            e = end[i]-1
            edge_pred[str((s,e))] = self.pBgV(samples[:,e]-samples[:,s],y[i]).mean()
        self.pred = edge_pred
        return edge_pred

    # def predict_val(self):
    #     samples = self.result.extract()['v']
    #     start = self.cycle_dat['src']
    #     end = self.cycle_dat['dst']
    #     y = self.cycle_dat['y']

    #     edge_pred = dict()
    #     for i in range(len(start)):
    #         s = start[i]-1
    #         e = end[i]-1
    #         edge_pred[str((s,e))] = (samples[:,e]-samples[:,s]).mean()
    #     self.pred = edge_pred
    #     return edge_pred
    
    def predict_val(self):
        if self.sample:
            samples = self.result.extract()['v']
        else:
            samples = self.result['v']
        start = self.cycle_dat['src']
        end = self.cycle_dat['dst']
        y = self.cycle_dat['y']

        edge_pred = dict()
        for i in range(len(start)):
            s = self.idx2node[start[i]-1]
            e = self.idx2node[end[i]-1]
            if not self.sample:
                edge_pred[(s,e)] = samples[end[i]-1]-samples[start[i]-1]
            else:
                edge_pred[(s,e)] = (samples[:,end[i]-1]-samples[:,start[i]-1]).mean()
        self.pred = edge_pred
        return edge_pred


    def auc(self,truth):
        order = truth.keys()
        edge_truth_arr = np.array([truth[k][0] for k in order])
        edge_pred_arr = np.array([self.pred[k] for k in order])
        return roc_auc_score(edge_truth_arr,edge_pred_arr)

    def auc_plot(self,truth):
        order = truth.keys()
        edge_truth_arr = np.array([truth[k][0] for k in order])
        edge_pred_arr = np.array([self.pred[k] for k in order])
        fpr, tpr, thresholds = roc_curve(edge_truth_arr, edge_pred_arr)
        plt.plot(fpr,tpr)
        plt.plot([0, 1], [0, 1], linestyle='--',color='r')

    def recall(self,truth,thres=0.5):
        order = truth.keys()
        edge_truth_arr = np.array([truth[k][0] for k in order])
        edge_pred_arr = np.array([self.pred[k] for k in order])
        pos_pred = edge_pred_arr[edge_truth_arr==1]
        plt.hist(pos_pred,density=True,color='r',histtype='step')
        print(f'TP/TP+FN: {(pos_pred>thres).sum()/len(pos_pred)}')

    def precision(self,truth,thres=0.5):
        order = truth.keys()
        edge_truth_arr = np.array([truth[k][0] for k in order])
        edge_pred_arr = np.array([self.pred[k] for k in order])
        print(f'TP/TP+FP: {precision_score(edge_truth_arr,edge_pred_arr>thres)}')
    
    def confusion_matrix(self,truth,thres=0.5,normalize=None):
        order = truth.keys()
        edge_truth_arr = np.array([truth[k][0] for k in order])
        edge_pred_arr = np.array([self.pred[k] for k in order])
        ConfusionMatrixDisplay(confusion_matrix(edge_truth_arr,edge_pred_arr>thres,normalize=normalize),['Good','Bad']).plot()
        return confusion_matrix(edge_truth_arr,edge_pred_arr>thres,normalize=normalize)
        
class Node_model_Uniform(Node_model):
    def __init__(self, prior=(-1,1), noise=(0,1)):
        cycle_code = """
            data {{
                int<lower=1> N;    // Total number of nodes
                int<lower=1> M;    // Total number of edges
                int<lower=1> src[M]; // Edge connectivity
                int<lower=1> dst[M]; 
                real y[M];         // measured edge data
            }}
            parameters {{
                vector<lower={pu},upper={ps}>[N] v;       // value at each node
            }}
            model {{
                v ~ uniform({pu}, {ps});
                for (e in 1:M) {{
                target += normal_lpdf( v[dst[e]] - v[src[e]] - y[e] | {gu}, {gs}); // bad noise assumption 
                }}
            }}
            """.format(pu=prior[0],ps=prior[1],gu=noise[0],gs=noise[1])
        self.prior = prior
        self.model = pystan.StanModel(model_code=cycle_code)


class Node_model_Gamma(Node_model):
    def __init__(self, prior=(2.,2.), noise=(0,1)):
        cycle_code = """
            data {{
                int<lower=1> N;    // Total number of nodes
                int<lower=1> M;    // Total number of edges
                int<lower=1> src[M]; // Edge connectivity
                int<lower=1> dst[M]; 
                real y[M];         // measured edge data
            }}
            parameters {{
                vector<lower=0>[N] v;       // value at each node
            }}
            model {{
                v ~ gamma({pu}, {ps});
                for (e in 1:M) {{
                target += normal_lpdf( v[dst[e]] - v[src[e]] - y[e] | {gu}, {gs}); // bad noise assumption 
                }}
            }}
            """.format(pu=prior[0],ps=prior[1],gu=noise[0],gs=noise[1])
        self.prior = prior
        self.model = pystan.StanModel(model_code=cycle_code)
