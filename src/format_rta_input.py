"""
    iterates over the million playlist dataset and outputs info
    about what is in there.
    THIS IS A MODIFICATION OF THE SCRIPT stats.py ORIGINALLY PROVIDED

    Usage:

        python format_rta_input.py path-to-mpd-data output-path
"""
import sys
import json
import re
import collections
import os
import datetime
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
import csv
import tqdm
import pickle
from collections import defaultdict
import argparse
from src.embeddings.model import MatrixFactorizationModel
from src.data_manager.data_manager import DataManager

total_playlists = 0
total_tracks = 0
unique_track_count = 0
tracks = set()
artists = set()
albums = set()
titles = set()
total_descriptions = 0
ntitles = set()
n_playlists = 1000000
n_tracks = 2262292
playlist_track = lil_matrix((n_playlists, n_tracks), dtype=np.int32) # to build interaction matrix of binary value
tracks_info = {} # to keep base infos on tracks
title_histogram = collections.Counter()
artist_histogram = collections.Counter()
track_histogram = collections.Counter()
last_modified_histogram = collections.Counter()
num_edits_histogram = collections.Counter()
playlist_length_histogram = collections.Counter()
num_followers_histogram = collections.Counter()
playlists_list = []
quick = False
max_files_for_quick_processing = 2


def process_mpd(raw_path, out_path):
    print("processing MPD")
    global playlists_list
    count = 0
    filenames = os.listdir(raw_path)
    for filename in tqdm.tqdm(sorted(filenames, key=str)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            playlists_list = []
            fullpath = os.sep.join((raw_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            process_info(mpd_slice["info"])
            for playlist in mpd_slice["playlists"]:
                process_playlist(playlist)
            count += 1
            seqfile = open('%s/playlists_seq.csv' % out_path, 'a', newline ='')
            with seqfile:
              write = csv.writer(seqfile) 
              write.writerows(playlists_list)  

            if quick and count > max_files_for_quick_processing:
                break

    show_summary()


def show_summary():
    print()
    print("number of playlists", total_playlists)
    print("number of tracks", total_tracks)
    print("number of unique tracks", len(tracks))
    print("number of unique albums", len(albums))
    print("number of unique artists", len(artists))
    print("number of unique titles", len(titles))
    print("number of playlists with descriptions", total_descriptions)
    print("number of unique normalized titles", len(ntitles))
    print("avg playlist length", float(total_tracks) / total_playlists)
    print()
    print("top playlist titles")
    for title, count in title_histogram.most_common(20):
        print("%7d %s" % (count, title))

    print()
    print("top tracks")
    for track, count in track_histogram.most_common(20):
        print("%7d %s" % (count, track))

    print()
    print("top artists")
    for artist, count in artist_histogram.most_common(20):
        print("%7d %s" % (count, artist))

    print()
    print("numedits histogram")
    for num_edits, count in num_edits_histogram.most_common(20):
        print("%7d %d" % (count, num_edits))

    print()
    print("last modified histogram")
    for ts, count in last_modified_histogram.most_common(20):
        print("%7d %s" % (count, to_date(ts)))

    print()
    print("playlist length histogram")
    for length, count in playlist_length_histogram.most_common(20):
        print("%7d %d" % (count, length))

    print()
    print("num followers histogram")
    for followers, count in num_followers_histogram.most_common(20):
        print("%7d %d" % (count, followers))


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def to_date(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")


def process_playlist(playlist):
    global total_playlists, total_tracks, total_descriptions, unique_track_count, playlists_list

    total_playlists += 1
    # print playlist['playlist_id'], playlist['name']

    if "description" in playlist:
        total_descriptions += 1

    titles.add(playlist["name"])
    nname = normalize_name(playlist["name"])
    ntitles.add(nname)
    title_histogram[nname] += 1

    playlist_length_histogram[playlist["num_tracks"]] += 1
    last_modified_histogram[playlist["modified_at"]] += 1
    num_edits_histogram[playlist["num_edits"]] += 1
    num_followers_histogram[playlist["num_followers"]] += 1
    playlist_id = playlist["pid"]
    playlist_track_count = 0
    playlist_seq = []
    for track in playlist["tracks"]:
      full_name = track["track_uri"].lstrip("spotify:track:")
      if full_name not in tracks_info :
        del track["pos"]
        tracks_info[full_name] = track
        unique_track_count += 1
        tracks_info[full_name]["id"] = unique_track_count - 1
        tracks_info[full_name]["count"] = 1
      elif playlist_track[playlist_id, tracks_info[full_name]["id"]] != 0 :
        # remove tracks that are already earlier in the playlist
        continue
      else :
        tracks_info[full_name]["count"] += 1
      total_tracks += 1
      albums.add(track["album_uri"])
      tracks.add(track["track_uri"])
      artists.add(track["artist_uri"])
      artist_histogram[track["artist_name"]] += 1
      track_histogram[full_name] += 1
      track_id = tracks_info[full_name]["id"]
      playlist_track_count += 1
      playlist_track[playlist_id, track_id] = playlist_track_count
      playlist_seq.append(str(track_id))
    playlists_list.append(playlist_seq)


def process_info(_):
    pass

def process_album_artist( tracks_info, out_path):
    artist_songs = defaultdict(list)  # a dict where keys are artist ids and values are list of corresponding songs
    album_songs = defaultdict(list)  # a dict where keys are album ids and values are list of corresponding songs
    song_album = np.zeros(n_tracks)  # a 1-D array where the index is the track id and the value is the album id
    song_artist = np.zeros(n_tracks)  # a 1-D array where the index is the track id and the value is the artist id
    album_ids = {}  # a dict where keys are album names and values are album ids
    artist_ids = {}  # a dict where keys are artist names and values are artist ids
    album_names = []  # a list where indices are album ids and values are album names
    artist_names = []  # a list where indices are artist ids and values are album names
    print("Processing albums and artists.")
    for d in tqdm.tqdm(tracks_info.values()):
        album_name = "%s by %s" % (d['album_name'], d['artist_name'])
        artist_name = d['artist_name']
        if album_name not in album_ids:
            album_id = len(album_names)
            album_ids[album_name] = album_id
            album_names.append(album_name)
        else:
            album_id = album_ids[album_name]
        song_album[d['id']] = album_id

        if artist_name not in artist_ids:
            artist_id = len(artist_names)
            artist_ids[artist_name] = artist_id
            artist_names.append(artist_name)
        else:
            artist_id = artist_ids[artist_name]
        song_artist[d['id']] = artist_id
        album_songs[album_id].append(d['id'])
        artist_songs[artist_id].append(d['id'])

    np.save('%s/song_album' % out_path, song_album)
    np.save('%s/song_artist' % out_path, song_artist)
    with open("%s/album_ids.pkl" % out_path, 'wb+') as f:
      pickle.dump(album_ids, f)

    with open("%s/artist_ids.pkl" % out_path, 'wb+') as f:
      pickle.dump(artist_ids, f)

    with open("%s/artist_songs.pkl" % out_path, 'wb+') as f:
      pickle.dump(artist_songs, f)

    with open("%s/album_songs.pkl" % out_path, 'wb+') as f:
      pickle.dump(album_songs, f)

    with open("%s/artist_names.pkl" % out_path, 'wb+') as f:
      pickle.dump(artist_names, f)

    with open("%s/album_names.pkl" % out_path, 'wb+') as f:
      pickle.dump(album_names, f)
    return

def create_initial_embeddings(data_manager):
    print("Creating initial song embeddings")
    mf_model = MatrixFactorizationModel(data_manager, retrain=True, emb_size=128)
    return

def create_side_embeddings(data_manager):
    buckets_dur = data_manager.get_duration_bucket(data_manager.song_duration)
    buckets_pop = data_manager.get_pop_bucket(data_manager.song_pop)
    buckets_dur_dict = {i: [] for i in range(40)}
    buckets_pop_dict = {i: [] for i in range(100)}
    print("Creating duration buckets")
    for ind, b in enumerate(buckets_dur):
        buckets_dur_dict[b].append(ind)
    print("Creating popularity buckets")
    for ind, b in enumerate(buckets_pop):
        buckets_pop_dict[b].append(ind)

    print([len(v) for k,v in buckets_pop_dict.items()])
    # Create metadata initial embedding
    song_embeddings = np.load(data_manager.song_embeddings_path)
    print("Creating album embeddings")
    alb_embeddings = np.asarray([song_embeddings[data_manager.album_songs[i]].mean(axis=0) for i in tqdm.tqdm(range(len(data_manager.album_songs)))])
    print("Creating artist embeddings")

    art_embeddings = np.asarray([song_embeddings[data_manager.artist_songs[i]].mean(axis=0) for i in tqdm.tqdm(range(len(data_manager.artist_songs)))])

    pop_embeddings = np.asarray(
        [song_embeddings[buckets_pop_dict[i]].mean(axis=0) for i in tqdm.tqdm(range(len(buckets_pop_dict)))])
    pop_embeddings[np.isnan(pop_embeddings)] = 0

    dur_embeddings = np.asarray(
        [song_embeddings[buckets_dur_dict[i]].mean(axis=0) for i in tqdm.tqdm(range(len(buckets_dur_dict)))])

    np.save(data_manager.album_embeddings_path, alb_embeddings)
    np.save(data_manager.artist_embeddings_path, art_embeddings)
    np.save(data_manager.pop_embeddings_path, pop_embeddings)
    np.save(data_manager.dur_embeddings_path, dur_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpd_path", type=str, required=False, default="../MPD/data",
                             help = "Path to MPD")
    parser.add_argument("--out_path", type=str, required=False, default="resources/data/rta_input",
                             help = "Path to rta input")
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs("resources/models", exist_ok=True)
    process_mpd(args.mpd_path, args.out_path)
    save_npz('%s/playlist_track.npz' % args.out_path, playlist_track.tocsr(False))
    with open('%s/tracks_info.json' % args.out_path, 'w') as fp:
      json.dump(tracks_info, fp, indent=4)
    process_album_artist(tracks_info, args.out_path)
    data_manager = DataManager()
    print(data_manager.binary_train_set)
    create_initial_embeddings(data_manager)
    create_side_embeddings(data_manager)


