import numpy as np
import argparse
from src.data_manager.data_manager import DataManager
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tqdm
from matplotlib.ticker import FormatStrFormatter

def confidence_interval(metrics):
    # Compute 95% confidence interval
    n = metrics.shape[0]
    std = metrics.std()
    return 1.96 * (std/np.sqrt(n))

def create_grouping_matrix():
    # Multiplying by this matrix gives a grouped average over 1000 rows.
    # Useful for averaging over test playlists with the same n_seed
    M = np.zeros((10000, 10))
    kernel = np.ones((1,1000))/ 1000
    for i in range(10):
      M[1000*i: 1000* (i+1), i] = kernel
    return M

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type = str, required = True,
                    help = "metric to compute")
    parser.add_argument("--recos_path", type = str, required = False,
                    help = "path to recos", default="resources/recos")
    parser.add_argument("--plots_path", type = str, required = False,
                    help = "path to plots", default="resources/plots")
    parser.add_argument("--models", type = str, required = False,
                    help = "comma separated names of models to evaluate", default="SKNN,VSKNN,STAN,VSTAN,MF-AVG,MF-CNN,MF-GRU,MF-Transformer,FM-Transformer,NN-Transformer")

    args = parser.parse_args()
    model_names = args.models.split(",")
    l = len(model_names)
    os.makedirs(args.plots_path, exist_ok=True)
    recos = [np.load(("%s/%s.npy") % (args.recos_path, m)) for m in model_names]
    sns.set()
    sns.set_palette("bright")
    cp = sns.color_palette()
    data_manager = DataManager()
    test_evaluator, test_dataloader = data_manager.get_test_data("test")
    M = create_grouping_matrix()
    if args.metric == "all":
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i in tqdm.tqdm(range(l)):
            m = model_names[i]
            rec = recos[i]
            precs = 100 * test_evaluator.compute_all_precisions(rec).dot(M)
            recalls = 100 * test_evaluator.compute_all_recalls(rec).dot(M)
            R_precs = 100 * test_evaluator.compute_all_R_precisions(rec).dot(M)
            ndcgs = 100 *test_evaluator.compute_all_ndcgs(rec).dot(M)
            clicks = test_evaluator.compute_all_clicks(rec).dot(M)
            norm_pop = 100 * test_evaluator.compute_norm_pop(rec).dot(M)
            sns.lineplot(ax=axes[0, 0], x=data_manager.N_SEED_SONGS, y=precs, label=m, markers=True, linewidth=2.0,
                         color=cp[i])
            sns.lineplot(ax=axes[0, 1], x=data_manager.N_SEED_SONGS, y=recalls, label=m, markers=True, linewidth=2.0,
                         color=cp[i])
            sns.lineplot(ax=axes[0, 2], x=data_manager.N_SEED_SONGS, y=R_precs, label=m, markers=True, linewidth=2.0,
                         color=cp[i])
            sns.lineplot(ax=axes[1, 0], x=data_manager.N_SEED_SONGS, y=ndcgs, label=m, markers=True, linewidth=2.0,
                         color=cp[i])
            sns.lineplot(ax=axes[1, 1], x=data_manager.N_SEED_SONGS, y=clicks, label=m, markers=True, linewidth=2.0,
                         color=cp[i])
            sns.lineplot(ax=axes[1, 2], x=data_manager.N_SEED_SONGS, y=norm_pop, label=m, markers=True, linewidth=2.0,
                         color=cp[i])

        handles, labels = axes[1, 2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='right', borderaxespad=0.3)
        for row in tqdm.tqdm(axes):
            for ax in row:
                ax.get_legend().remove()
                ax.set_xticks(data_manager.N_SEED_SONGS)
                ax.tick_params(axis="both", labelsize=14)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[0, 0].set_ylabel("Precision (in %)", {"size": 14, 'weight': 'bold'})
        axes[0, 1].set_ylabel("Recall (in %)", {"size": 14, 'weight': 'bold'})
        axes[0, 2].set_ylabel("R-Precision (in %)", {"size": 14, 'weight': 'bold'})
        axes[1, 0].set_ylabel("NDCG (in %)", {"size": 14, 'weight': 'bold'})
        axes[1, 1].set_ylabel("Clicks (in number)", {"size": 14, 'weight': 'bold'})
        axes[1, 2].set_ylabel("Popularity (in %)", {"size": 14, 'weight': 'bold'})
        fig.text(0.5, 0.04, 'Number of seed songs', ha='center', size=18, weight='bold')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.25,
                            hspace=0.15)
        plt.savefig("all_results.pdf", bbox_inches="tight")
    else:
        for i in range(l):
            m = model_names[i]
            rec = recos[i]

            if args.metric== "coverage": # coverage can not be averaged so no confidence interval
                metrics = test_evaluator.compute_cov(rec)
                print("%s: %.4f" % (m, np.mean(metrics)))
            else:
                if args.metric == "recall":
                    metrics = test_evaluator.compute_all_recalls(rec)
                if args.metric == "ndcg":
                    metrics = test_evaluator.compute_all_ndcgs(rec)
                if args.metric == "clicks":
                    metrics = test_evaluator.compute_all_clicks(rec)
                if args.metric == "precision":
                    metrics = test_evaluator.compute_all_precisions(rec)
                if args.metric == "r-precision":
                    metrics = test_evaluator.compute_all_R_precisions(rec)
                if args.metric== "popularity":
                    metrics = test_evaluator.compute_norm_pop(rec)
                alpha = confidence_interval(metrics)
                lower = np.mean(metrics) - alpha
                upper = np.mean(metrics) + alpha
                groups = metrics.dot(M)
                sns.lineplot(x=data_manager.N_SEED_SONGS, y=groups, label=m, markers=True, linewidth=1.1, color=cp[i])
                print("%s: %.4f-%.4f" % (m, np.mean(metrics), alpha))
                print((metrics == 0).sum())
                plt.xlabel("number of seed tracks", {"size": 14, 'weight': 'bold'})
                plt.ylabel(args.metric, {"size": 14, 'weight': 'bold'})
                plt.xticks(fontsize=12, ticks=data_manager.N_SEED_SONGS)
                plt.yticks(fontsize=12)
                plt.legend(loc="best")
                plt.savefig("%s/%s.pdf" % (args.plots_path, args.metric), bbox_inches = "tight")
                plt.show()