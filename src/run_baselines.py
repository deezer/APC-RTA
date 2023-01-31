import numpy as np
from src.data_manager.data_manager import DataManager
from src.utils import array_mapping
from src.evaluator import Evaluator
import json, argparse
import pandas as pd
from src.baselines.vsknn import VMContextKNN
from src.baselines.sknn import ContextKNN
from src.baselines.vstan import VSKNN_STAN
from src.baselines.stan import STAN
import tqdm
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True,
                    help = "Name of model to train")
    parser.add_argument("--data_path", type = str, required = False,
                    help = "path to data", default="resources/data/baselines")
    parser.add_argument("--params_file", type = str, required = False,
                    help = "file for parameters", default="resources/params/best_params_baselines.json")
    args = parser.parse_args()
    with open(args.params_file, "r") as f:
      p = json.load(f)

    tr_params = p[args.model_name]
    df_train = pd.read_hdf("%s/df_train_for_test" % args.data_path)
    if args.model_name == "VSKNN":
        knnModel = VMContextKNN(k=tr_params["k"], sample_size=tr_params["n_sample"], weighting=tr_params["w"], weighting_score=tr_params["w_score"],  idf_weighting=tr_params["idf_w"])

    if args.model_name == "SKNN":
        knnModel = ContextKNN(k=tr_params["k"], sample_size=tr_params["n_sample"], similarity= tr_params["s"])

    if args.model_name == "VSTAN":
        df_train["Time"] = df_train["Time"] / 1000 # necessary to avoid overflow
        knnModel = VSKNN_STAN(k=tr_params["k"], sample_size=tr_params["n_sample"], lambda_spw=tr_params["sp_w"],  lambda_snh=tr_params["sn_w"], lambda_inh=tr_params["in_w"])

    if args.model_name == "STAN":
        knnModel = STAN(k=tr_params["k"], sample_size=tr_params["n_sample"], lambda_spw=tr_params["sp_w"],  lambda_snh=tr_params["sn_w"], lambda_inh=tr_params["in_w"])
    data_manager = DataManager()
    last_item = df_train[df_train.SessionId.isin(data_manager.test_indices)].sort_values("Time", ascending=False).groupby("SessionId", as_index=False).first()
    all_tids = np.arange(data_manager.n_tracks)
    unknown_tracks = list(set(np.arange(data_manager.n_tracks)) - set(df_train.ItemId.unique()))

    gt_test = []
    for i in DataManager.N_SEED_SONGS:
      gt_test += data_manager.ground_truths["test"][i]

    n_recos = 500
    test_to_last = array_mapping(data_manager.test_indices, last_item.SessionId.values)

    start_fit = time.time()
    print("Start fitting knn model")
    knnModel.fit(df_train)
    end_fit = time.time()
    print("Training done in %.2f seconds" % (end_fit - start_fit))
    print("Start predicting knn model")
    recos_knn = np.zeros((10000, 500))
    for i, (pid, tid, t) in tqdm.tqdm(enumerate(last_item[["SessionId", "ItemId", "Time"]].values)):
        pl_tracks = df_train[df_train.SessionId == pid].ItemId.values
        scores = knnModel.predict_next(pid, tid, all_tids)
        scores[pl_tracks] = 0
        scores[unknown_tracks] = 0
        recos_knn[i] = np.argsort(-scores)[:500]

    recos_sorted = recos_knn[test_to_last] # same order as rta models evaluator
    end_predict = time.time()
    print("Training done in %.2f seconds" % (end_predict - end_fit))
    np.save("resources/recos/%s" %args.model_name, recos_sorted)