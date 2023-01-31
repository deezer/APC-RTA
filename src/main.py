import json, argparse, os, time, datetime

from src.data_manager.data_manager import DataManager

from src.rta.utils import get_device
from src.rta.rta_model import RTAModel
from src.rta.aggregator.gru import GRUNet
from src.rta.aggregator.cnn import GatedCNN
from src.rta.aggregator.decoder import DecoderModel
from src.rta.aggregator.base import AggregatorBase
from src.rta.representer.base_representer import BaseEmbeddingRepresenter
from src.rta.representer.fm_representer import FMRepresenter
from src.rta.representer.attention_representer import AttentionFMRepresenter
import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type = str, required = True,
                    help = "Name of model to train")
  parser.add_argument("--params_file", type = str, required = False,
                    help = "File for hyperparameters", default = "resources/params/best_params_rta.json")
  parser.add_argument("--recos_path", type = str, required = False,
                    help = "Path to save recos", default = "resources/recos")
  parser.add_argument("--models_path", type = str, required = False,
                    help = "Path to save models", default = "resources/models")
  args = parser.parse_args()

  data_manager = DataManager()
  with open(args.params_file, "r") as f:
    p = json.load(f)

  tr_params = p[args.model_name]
  if args.model_name == "MF-GRU":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize GRU")
    aggregator = GRUNet(tr_params['d'], tr_params['h_dim'], tr_params['d'], tr_params['n_layers'], tr_params['drop_p'])

  if args.model_name == "MF-CNN":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize Gated-CNN")
    aggregator = GatedCNN(tr_params['d'], tr_params['n_layers'], tr_params['kernel_size'], tr_params['conv_size'], tr_params['res_block_count'], k_pool=tr_params['k_pool'], drop_p=tr_params['drop_p']).to(get_device())

  if args.model_name == "MF-AVG":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize vanilla matrix factorization")
    aggregator = AggregatorBase()

  if args.model_name == "MF-Transformer":
    print("Initialize Embeddings")
    representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
    print("Initialize Decoder")
    aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

  if args.model_name == "FM-Transformer":
    print("Initialize Embeddings")
    representer = FMRepresenter(data_manager, tr_params['d'])
    print("Initialize Decoder")
    aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

  if args.model_name == "NN-Transformer":
    print("Initialize Embeddings")
    representer = AttentionFMRepresenter(data_manager, emb_dim=tr_params['d'], n_att_heads=tr_params['n_att_heads'], n_att_layers=tr_params["n_att_layers"], dropout_att=tr_params["drop_att"])
    print("Initialize Decoder")
    aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

  rta_model = RTAModel(data_manager, representer, aggregator, training_params = tr_params).to(get_device())
  print("Train model %s" % args.model_name)
  savePath = "%s/%s" % (args.models_path, args.model_name)
  start_fit = time.time()
  rta_model.run_training(tuning=False, savePath=savePath)
  end_fit = time.time()
  print("Model %s trained in %s " % (args.model_name, str(end_fit - start_fit)))
  test_evaluator, test_dataloader = data_manager.get_test_data("test")
  recos = rta_model.compute_recos(test_dataloader)
  end_predict = time.time()
  print("Model %s inferred in %s " % (args.model_name, str(end_predict - end_fit)))
  os.makedirs(args.recos_path, exist_ok=True)
  np.save("%s/%s" % (args.recos_path, args.model_name), recos)
