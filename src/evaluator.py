from math import log2, floor
import torch
import numpy as np
class Evaluator():
  """ A class dedicated to computing metrics given a ground truth"""
  def __init__(self, data_manager, gt, n_recos = 500):
    self.gt = gt
    self.test_size = len(self.gt)
    self.n_recos = n_recos
    self.song_pop = torch.LongTensor(data_manager.song_pop)
    self.song_artist = data_manager.song_artist

  def compute_all_recalls(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_recall(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_RR(self, recos, gt):
    if len(gt) ==0:
      return 1
    return max([1/(i+1) for i in range(len(recos)) if recos[i] in gt] + [0])

  def compute_all_RRs(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_RR(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_recall(self, recos, gt):
    R = len(gt)
    if R == 0:
      return 1
    return len(set(recos).intersection(gt)) / R

  def compute_single_precision(self, recos, gt):
    R = len(gt)
    if R == 0:
      return 1
    return len(set(recos).intersection(gt)) / len(recos)

  def compute_all_precisions(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_precision(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_R_precision(self, recos, gt):
    R = len(gt)
    if R == 0:
      return 1
    song_score = len(set(recos[:R]).intersection(gt))
    artist_score = len(set(self.song_artist[recos.astype(np.int64)].astype(np.int64)).intersection(set(self.song_artist[list(gt)].astype(np.int64))))
    return (song_score + 0.25*artist_score) / R

  def compute_all_R_precisions(self, recos): #TODO : add artist share
    n = len(self.gt)
    return np.array([self.compute_single_R_precision(recos[i], self.gt[i]) for i in range(n)])

  def compute_single_click(self, recos, gt):
    n_recos = recos.shape[0]
    if len(gt) ==0:
      return 1
    return next((floor(i/10) for i in range(n_recos) if recos[i] in gt), 51)

  def compute_all_clicks(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_click(recos[i], self.gt[i]) for i in range(n)])

  def compute_pop(self, recos):
    s = recos.shape
    pop = torch.gather(self.song_pop, 0, torch.LongTensor(recos.reshape(-1))).reshape(s)
    return pop

  def compute_norm_pop(self, recos):
    pop = self.compute_pop(recos)
    max_pop = self.song_pop.max()
    min_pop = self.song_pop.min()
    return np.array((pop - min_pop) / (max_pop -min_pop)).mean(axis=1)

  def compute_cov(self, recos):
    return len(np.unique(recos)) / len(self.song_pop)

  def compute_single_dcg(self, recos, gt):
    if len(gt) ==0:
      return 1
    return sum([1/log2(i+2) for i in range(len(recos)) if recos[i] in gt])

  def compute_single_ndcg(self, recos, gt):
    dcg = self.compute_single_dcg(recos, gt)
    idcg = self.compute_single_dcg(list(gt), gt)
    return dcg/idcg
  
  def compute_all_ndcgs(self, recos):
    n = len(self.gt)
    return np.array([self.compute_single_ndcg(recos[i], self.gt[i]) for i in range(n)])