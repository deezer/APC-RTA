from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz
import numpy as np
import os
import json
import operator
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import random
import pickle
import math
from src.evaluator import Evaluator
from torch.utils.data import DataLoader

class DataManager():
  '''A class managing data access for models/evaluators.
      - Reads raw files
      - Can split between test/val/test, either by using the one in resources or computing a new one
      - Offers different representation for playlist-track information (sequential and matricial)
      - Manages side information associated with each track (album/artist/popularity bucket/duration bucket)
      - Gives access to embeddings'''
  N_SEED_SONGS = range(1,11) # possible configurations for evaluation

  def __init__(self, foldername = "resources/data/", test_size=10000, min_songs_test=10, resplit=False, dim=128):
    self.foldername = foldername 
    self.test_size = test_size
    self.min_songs_test = min_songs_test
    self.load_playlist_track()
    self.song_embeddings_path = '%s/embeddings/song_embeddings_%d.npy' % (self.foldername, dim)
    self.album_embeddings_path = '%s/embeddings/alb_embeddings_%d.npy' % (self.foldername, dim)
    self.artist_embeddings_path = '%s/embeddings/art_embeddings_%d.npy' % (self.foldername, dim)
    self.pop_embeddings_path = '%s/embeddings/pop_embeddings_%d.npy' % (self.foldername, dim)
    self.dur_embeddings_path = '%s/embeddings/dur_embeddings_%d.npy' % (self.foldername, dim)

    self.load_track_info()
    self.load_metadata()

    self.n_playlists = 10**6
    self.n_tracks = 2262292
    self.train_indices = self.get_indices("train", resplit=resplit)
    self.val_indices = self.get_indices("val")
    self.test_indices = self.get_indices("test")
    self.ground_truths = {}
    self.ground_truths_first = {}

    self.seed_tracks = {}
    tmp = [self.get_ground_truth("val", n_start_songs = i) for i in DataManager.N_SEED_SONGS]
    self.seed_tracks["val"] = {i:tmp[ind][0] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
    self.ground_truths["val"] = {i:tmp[ind][1] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
    self.ground_truths_first["val"] = {i:tmp[ind][2] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
    tmp = [self.get_ground_truth("test", n_start_songs = i) for i in DataManager.N_SEED_SONGS]
    self.seed_tracks["test"] = {i:tmp[ind][0] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
    self.ground_truths["test"] = {i:tmp[ind][1] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
    self.ground_truths_first["test"] = {i:tmp[ind][2] for ind,i in enumerate(DataManager.N_SEED_SONGS)}
    self.binary_train_set = self.get_train_set()
    self.prepare_charts()
    #self.val_input = self.binary_val_set.indices.reshape((self.binary_val_set.shape[0], self.min_songs_test))

  def load_playlist_track(self):
    self.playlist_track = load_npz("%s/rta_input/playlist_track.npz" % self.foldername)
  
  def load_playlist_artist(self):
    self.playlist_artist = load_npz("%s/rta_input/playlist_artist.npz" % self.foldername)

  def load_playlist_album(self):
    self.playlist_album = load_npz("%s/rta_input/playlist_album.npz" % self.foldername)

  def load_metadata(self):
    # load metadata information such as album/artist/duration/popularity

    self.song_album = np.load("%s/rta_input/song_album.npy" % self.foldername)
    self.song_artist = np.load("%s/rta_input/song_artist.npy" % self.foldername)
    song_infos_sorted = sorted([(info['id'], info['count'], info['duration_ms']) for info in self.tracks_info.values()])
    self.song_pop = [c[1] for c in song_infos_sorted]
    self.song_duration = [c[2] for c in song_infos_sorted]

    with open("%s/rta_input/album_ids.pkl" % self.foldername, 'rb+') as f:
      self.album_ids = pickle.load(f)

    with open("%s/rta_input/artist_ids.pkl" % self.foldername, 'rb+') as f:
      self.artist_ids = pickle.load(f)

    with open("%s/rta_input/artist_songs.pkl" % self.foldername, 'rb+') as f:
      self.artist_songs = pickle.load(f)
  
    with open("%s/rta_input/album_songs.pkl" % self.foldername, 'rb+') as f:
      self.album_songs = pickle.load(f)

    with open("%s/rta_input/artist_names.pkl" % self.foldername, 'rb+') as f:
      self.artist_names = pickle.load(f)

    with open("%s/rta_input/album_names.pkl" % self.foldername, 'rb+') as f:
      self.album_names = pickle.load(f)

  def load_track_info(self):
    with open("%s/rta_input/tracks_info.json" % self.foldername) as f :
      self.tracks_info = json.load(f)

  def get_duration_bucket(self, x):
    # Songs are assigned a duration bucket depending on their length.
    # A new bucket is created every 30 seconds, except for all songs longer than 20 minutes that all belong to the
    # last bucket.
    MAX_DURATION = 1200000 # all songs longer than 20 minutes belong to the last bucket
    if (type(x) == torch.Tensor):
      buckets = torch.div(40 * x, MAX_DURATION, rounding_mode='trunc')
    else :
      buckets = (40 * np.array(x) / MAX_DURATION).astype(int)
    buckets = buckets * (buckets > 0) # low values are set to 0
    buckets = buckets * (buckets < 40) + 39 * (buckets >= 40) # high values are set to 39
    return buckets

  def get_pop_bucket(self, x):
    # Songs are assigned a duration bucket depending on their number of occurences in the training set length.
    # 100 buckets are created with logarithmic intervals of number of occurences.
    x[x==0] = 1
    MAX_POP = 45394 # all songs more frequent than this belong to the last bucket
    if (type(x) == torch.Tensor):
      buckets = 1 + torch.div(100 * torch.log(x/2), np.log(MAX_POP/2), rounding_mode='trunc') # 2 -> 1 and MAX_POP -> 101
    else:
      buckets = 1 + (100 * (np.log(x) - np.log(2)) / (np.log(MAX_POP) - np.log(2))).astype(int)
    buckets = buckets * (buckets > 0) # low values are set to 0
    buckets = buckets * (buckets < 100) + 99 * (buckets >= 100)  # high values are set to 99
    return buckets
  
  def prepare_charts(self):
    # Sort tracks by their number of occurences in the training set.
    self.ordered_tracks = [e[0] for e in sorted({v["id"]:v["count"] for k,v in self.tracks_info.items()}.items(), key=operator.itemgetter(1), reverse=True)]
    self.ordered_tracks.insert(0, self.n_tracks)
    self.tracks_rank = np.zeros(self.n_tracks + 1, dtype=np.int32)
    for i,t in enumerate(self.ordered_tracks):
      self.tracks_rank[t] = i
    self.ordered_tracks = np.array(self.ordered_tracks)
  
  def split_sets(self):
    # split MPD between train/validation/test

    playlist_track_csc = self.playlist_track.tocsc()
    rng = np.random.default_rng()
    candidate_indices = rng.choice(list(set(playlist_track_csc.indices[playlist_track_csc.data > 2*self.min_songs_test])), 2*self.test_size, replace = False) # find all playlists that have at least 10 songs
  
    test_indices = candidate_indices[:self.test_size] 
    val_indices = candidate_indices[self.test_size:]
    train_indices = [i for i in range(self.n_playlists) if i not in candidate_indices]

    np.save('%s/dataset_split/train_indices' % (self.foldername), train_indices)
    np.save('%s/dataset_split/val_indices' % (self.foldername), val_indices)
    np.save('%s/dataset_split/test_indices' % (self.foldername), test_indices)

  def get_indices(self, set_name, resplit = False):
    if resplit:
      self.split_sets()
    return np.load("%s/dataset_split/%s_indices.npy" % (self.foldername, set_name))
      
  def get_valid_playlists(self, train_indices, test_indices):
    # Remove playlists in test set that contain songs with no occurences in the train set
    train_tracks = set(self.playlist_track[train_indices].indices)
    test_tracks = set(self.playlist_track[test_indices].indices)
    test_size = len(test_indices)
    invalid_tracks = test_tracks - train_tracks
    invalid_positions = set()
    v = self.playlist_track[test_indices].tocsc()
    for i in invalid_tracks:
      invalid_positions = invalid_positions.union(set(v.indices[v.indptr[i]:v.indptr[i+1]]))
    valid_positions = np.array(sorted([p for p in range(test_size) if p not in invalid_positions]))
    return test_indices[valid_positions]
  
  def get_ground_truth(self, set_name, binary = True, resplit=False, n_start_songs = False):
    # Get songs that are at the end of playlists from the test set
    if not n_start_songs:
      n_start_songs = self.min_songs_test
    indices = self.get_indices(set_name, resplit)
    data = self.playlist_track[indices[1000 * (n_start_songs-1): 1000 * n_start_songs]] # select 1000 tracks for this configuration
    ground_truth_array = data.multiply(data > n_start_songs)
    ground_truth_first = data.multiply(data == (n_start_songs+1)) # first_track of ground_truth
    start_data = data - ground_truth_array
    if binary:
      start_data = 1 * (start_data > 0)
    ground_truth_list = []
    ground_truth_list_first = []
    for i in range(data.shape[0]):
      ground_truth_list.append(set(ground_truth_array.indices[ground_truth_array.indptr[i]:ground_truth_array.indptr[i+1]]))
      ground_truth_list_first.append(set(ground_truth_first.indices[ground_truth_first.indptr[i]:ground_truth_first.indptr[i+1]]))
    return start_data, ground_truth_list, ground_truth_list_first

  def get_train_set(self, binary = True, resplit=False):
    train_indices = self.get_indices("train", resplit)
    train_set = self.playlist_track[train_indices]
    if binary :
      train_set = 1 * (train_set > 0)
    return train_set

  def get_test_data(self, mode, n_recos=500, test_batch_size=50):
    gt_test = [] 
    for i in DataManager.N_SEED_SONGS:
      gt_test += self.ground_truths[mode][i]
    test_evaluator = Evaluator(self, gt=np.array(gt_test), n_recos=n_recos)
    if mode == "test":
      test_dataset = EvaluationDataset(self, self.test_indices)
    else:
      test_dataset = EvaluationDataset(self, self.val_indices)
    test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=0)

    return test_evaluator, test_dataloader
  
class negative_sampler(object):
  """A class to speed up negative sampling. Instead of sampling uniformly at every call,
  the whole list of tracks is shuffled once then read by chunk. When the end of the list
  is reached, it is shuffled again to start reading from the beginning etc..."""

  def __init__(self, n_max):
    self.n_max = n_max
    self.current_n = 0
    self.values = np.arange(n_max)
    np.random.shuffle(self.values)
  def __iter__(self):
    return self

  def __next__(self):
    return self.next()
  
  def next(self, size=1):
    if self.current_n + size >= self.n_max:
      np.random.shuffle(self.values)
      self.current_n = 0

    neg_samples = self.values[self.current_n:self.current_n+size]
    self.current_n = self.current_n+size
    return neg_samples

class SequentialTrainDataset(Dataset):
    # This class is used to load the training set. If a playlist is shorter than max_size = 50 song, select entire playmist
    # and pad later. Otherwise, randomly select a subplaylist of 50 consecutive tracks.
    def __init__(self, data_manager, indices, max_size=50, n_pos=3, n_neg=10):
      self.max_size = max_size
      self.n_pos = n_pos
      self.n_neg = n_neg
      self.data = data_manager.playlist_track[indices]
      self.neg_generator = negative_sampler(data_manager.n_tracks + 1)

    def __getitem__(self, index):
      A = self.data[index].indices + 1
      B = self.data[index].data
      seq = np.array([x for y, x in sorted(zip(B, A))])  # sort by position in the playlist, but keep song indices
      l = len(seq)
      if l <= self.max_size:
        X = seq
      else:
        start = np.random.randint(0, l - (self.max_size))
        X = seq[start:start + self.max_size]
      y_neg = self.sample_except_with_generator(self.n_neg, seq)
      return torch.LongTensor(X), torch.LongTensor(y_neg)

    def __len__(self):
      return self.data.shape[0]

    def sample_except_with_generator(self, n_samples, excluded_values):
      # Sample negative examples but remove songs that appear in the playlist
      l = len(excluded_values)
      raw_samples = self.neg_generator.next(n_samples)
      diff = set(raw_samples).difference(excluded_values)
      while (len(diff) < n_samples):
        l_res = n_samples - len(diff)
        diff = diff.union(set(self.neg_generator.next(l_res)).difference(excluded_values))
      return list(diff)


# Test : for each row, split text, convert to int, select 5 first tracks. Predict following tracks
class EvaluationDataset(Dataset):
    # This class is used to load either the validation or test set.
    # for each playlist, X is the beginning and Y is the end.
    def __init__(self, data_manager, indices):
      self.data = data_manager.playlist_track[indices]
    def __getitem__(self, index):
        n_seed = math.floor(index/1000) + 1 # select 1000 per value of n_seed
        X = self.data[index].indices + 1
        Y = self.data[index].data
        return np.array([x for y,x in sorted(zip(Y,X))][:n_seed])
    def __len__(self):
        return self.data.shape[0]

def pad_collate(batch):
    # Defines how playlists of different sizes should be turned into a batch using padding
    (xx, yy_neg) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.stack(list(yy_neg)), torch.LongTensor(list(x_lens))