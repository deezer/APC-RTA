import numpy as np
import pandas as pd
import tqdm
import os
import sys
from src.data_manager.data_manager import DataManager
import json
import math
import argparse

def process_playlists(mpd_path, out_path):
    count = 0
    filenames = os.listdir(mpd_path)
    all_data = []
    for filename in tqdm.tqdm(sorted(filenames)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((mpd_path, filename))
            with open(fullpath) as f:
                js = f.read()
            mpd_slice = json.loads(js)
            data = mpd_slice['playlists']
            for d in data:
                d.pop("tracks")
            all_data += data
    playlist_data = pd.DataFrame(all_data)
    playlist_data = playlist_data.drop("description", axis=1)
    playlist_data.to_hdf("%s/playlist_data" % out_path , "abc")

def prepare_knn_data(data_manager, data_path):
# prepare input for knn models
  data_manager.load_playlist_track()
  pt = data_manager.playlist_track
  n_int = len(pt.data)
  df_data = np.zeros((n_int, 3), dtype=np.int64)
  for pid in tqdm.tqdm(range(data_manager.n_playlists)):
    df_data[pt.indptr[pid]:pt.indptr[pid+1],0] = pid
    df_data[pt.indptr[pid]:pt.indptr[pid+1], 1] = pt.indices[pt.indptr[pid]:pt.indptr[pid+1]]
    df_data[pt.indptr[pid]:pt.indptr[pid+1], 2] = pt.data[pt.indptr[pid]:pt.indptr[pid+1]]

  cols = ["SessionId", "ItemId", "Pos"]
  df_data = pd.DataFrame(df_data, columns=cols)
  playlist_data = pd.read_hdf("%s/playlist_data" % data_path)
  df_data = df_data.merge(playlist_data[["pid", "modified_at"]].rename(columns={"pid":"SessionId"}), on="SessionId")
  df_data = df_data.merge(pd.DataFrame(data_manager.tracks_info.values())[["id", "duration_ms"]].rename(columns={'id':"ItemId"}), on="ItemId")
  df_data.sort_values(["SessionId", "Pos"], ascending = [True,  False], inplace=True)
  df_data["duration_sum"] = df_data.groupby('SessionId')["duration_ms"].cumsum()
  df_data["Time"] = df_data["modified_at"] - df_data["duration_sum"]
  return df_data

def prepare_val_input(data_manager, df_data):
  df_train = df_data[df_data.SessionId.isin(data_manager.train_indices)]
  df_val = df_data[df_data.SessionId.isin(data_manager.val_indices)]
  for i,idx in tqdm.tqdm(enumerate(data_manager.val_indices)):
    cat = math.floor(i / 1000) + 1
    df_val = df_val[~((df_val.SessionId == idx) & (df_val.Pos > cat))]
  df_train = pd.concat([df_train, df_val])
  return df_train

def prepare_test_input(data_manager, df_data):
  df_train = df_data[~df_data.SessionId.isin(data_manager.test_indices)]
  df_test = df_data[df_data.SessionId.isin(data_manager.test_indices)]
  for i,idx in tqdm.tqdm(enumerate(data_manager.test_indices)):
    cat = math.floor(i / 1000) + 1
    df_test = df_test[~((df_test.SessionId == idx) & (df_test.Pos > cat))]
  df_train = pd.concat([df_train, df_test])
  return df_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpd_path", type=str, required=False, default="../MPD/data",
                        help = "Path to MPD")
    parser.add_argument("--out_path", type=str, required=False, default="resources/data/baselines",
                        help = "Path to baselines input")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    data_manager = DataManager()
    process_playlists(args.mpd_path, args.out_path)
    df_data = prepare_knn_data(data_manager, args.out_path)
    df_data.to_hdf("%s/df_data" % args.out_path, "abc")
    df_train_tune = prepare_val_input(data_manager, df_data)
    df_train_tune.to_hdf("%s/df_train_for_val" % args.out_path, "abc")
    df_train_final = prepare_test_input(data_manager, df_data)
    df_train_final.to_hdf("%s/df_train_for_test" % args.out_path, "abc")