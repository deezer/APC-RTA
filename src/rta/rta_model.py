import torch
import torch.nn.functional as F
import numpy as np
from src.rta.utils import padded_avg, get_device
from src.data_manager.data_manager import SequentialTrainDataset, pad_collate
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import tqdm
import time
from torch import Tensor
from torch.nn.functional import log_softmax

class RTAModel(torch.nn.Module):
    """ The main class for creating RTA models. Each consist fo the combination of a Representer with an Aggregator,
     which are jointly trained by iterating over the training set (using the DataManager)"""
    def __init__(self,
               data_manager,
               representer,
               aggregator,
               training_params = {}):
      super(RTAModel, self).__init__()
      self.data_manager = data_manager
      self.representer = representer
      self.aggregator = aggregator
      self.training_params = training_params

    def chose_negative_examples(self, X_pos_rep, x_neg, pad_mask):
        # Negative examples are partly made of hard negatives and easy random negatives
        X_neg_rep = self.representer(x_neg)
        easy_neg_rep = X_neg_rep[:,:self.training_params['n_easy'],...]

        # draw hard negatives using nearst neighbours in the first layer song embedding space
        X_rep_avg = padded_avg(X_pos_rep, ~pad_mask)
        neg_prods = torch.diagonal(X_neg_rep.matmul(X_rep_avg.T), dim1=2, dim2=0).T
        top_neg_indices = torch.topk(neg_prods, k=self.training_params['n_hard'], dim=1)[1]
        hard_indices = torch.gather(x_neg, 1, top_neg_indices)

        hard_neg_rep = self.representer((hard_indices))
        X_neg_final = torch.cat([easy_neg_rep, hard_neg_rep], dim=1)
        return X_neg_final

    def compute_pos_loss_batch(self, X_agg, Y_pos_rep, pad_mask):
        # The part of the loss that concerns positive examples
        pos_prod = torch.sum(X_agg * Y_pos_rep, axis = 2).unsqueeze(2)
        pos_loss = padded_avg(-F.logsigmoid(pos_prod), ~pad_mask).mean()
        return pos_loss

    def compute_neg_loss_batch(self, X_agg, X_neg_rep, pad_mask):
        # The part of the loss that concerns positive examples
        X_agg_mean = padded_avg(X_agg, ~pad_mask)
        neg_prod = X_neg_rep.matmul(X_agg_mean.transpose(0,1)).transpose(1,2).diagonal().transpose(0,1)
        neg_loss = torch.mean(-F.logsigmoid(-neg_prod))
        return neg_loss

    def compute_loss_batch(self, x_pos, x_neg):
        # Computes the entirety of the loss for a batch
        pad_mask = (x_pos == 0).to(get_device())

        X_pos_rep = self.representer(x_pos)
        input_rep = X_pos_rep[:,:-1,:] # take all elements of each sequence except the last
        Y_pos_rep = X_pos_rep[:,1:,:]  # take all elements of each sequence except the first

        X_agg = self.aggregator.aggregate(input_rep, pad_mask[:,:-1])
        X_neg_rep = self.chose_negative_examples(X_agg, x_neg, pad_mask[:,1:])

        pos_loss = self.compute_pos_loss_batch(X_agg, Y_pos_rep, pad_mask[:,1:])
        neg_loss = self.compute_neg_loss_batch(X_agg, X_neg_rep, pad_mask[:,1:])
        loss = pos_loss + neg_loss
        return loss

    def prepare_training_objects(self, tuning=False):
        # Prepare the optimizer, the scheduler and the data_loader that will be used for training
        optimizer = torch.optim.SGD(self.parameters(), lr= self.training_params['lr'], weight_decay=self.training_params['wd'], momentum=self.training_params['mom'], nesterov=self.training_params['nesterov'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.training_params['patience'], gamma=self.training_params['factor'], last_epoch=- 1, verbose=False)
        if tuning:
            train_indices = self.data_manager.train_indices
        else:
            train_indices = np.concatenate((self.data_manager.train_indices, self.data_manager.val_indices))
        train_dataset = SequentialTrainDataset(self.data_manager, train_indices, max_size=self.training_params['max_size'], n_neg=self.training_params['n_neg'])
        train_dataloader = DataLoader(train_dataset, batch_size = self.training_params['batch_size'], shuffle=True, collate_fn=pad_collate, num_workers=0)
        return optimizer, scheduler, train_dataloader

    def compute_recos(self, test_dataloader, n_recos=500):
        # Compute recommendations for playlist of the validation or test sel
        dev = get_device()
        n_p = len(test_dataloader.dataset)
        with torch.no_grad():
        self.eval()
        recos = np.zeros((n_p, n_recos))
        current_batch = 0
        all_rep = self.representer.compute_all_representations()
        for X in test_dataloader:
          X = X.long().to(dev)
          bs = X.shape[0]
          seq_len = X.shape[1]
          X_rep = self.representer(X)
          X_agg = self.aggregator.aggregate_single(X_rep, torch.zeros((bs, seq_len)).to(dev))
          scores = X_agg.matmul(all_rep[1:-1].T)
          scores = scores.scatter(1, X.to(dev) - 1, value = - 10**3) # make sure songs in the seed are not recommended
          coded_recos = torch.topk(scores, k =n_recos, dim=1)[1].cpu().long()
          recos[current_batch * test_dataloader.batch_size: current_batch * test_dataloader.batch_size + bs] = coded_recos
          current_batch+=1
        self.train()
        return recos

    def run_training(self, tuning=False, savePath=False):
        # Runs the training loop of the RTAModel
      if tuning :
        test_evaluator, test_dataloader = self.data_manager.get_test_data("val")
      else :
        test_evaluator, test_dataloader = self.data_manager.get_test_data("test")
      optimizer, scheduler, train_dataloader = self.prepare_training_objects(tuning)
      batch_ct = 0
      print_every = False
      if "step_every" in self.training_params.keys():
        print_every = True
      start = time.time()
      if savePath:
        torch.save(self, savePath)
      for epoch in range(self.training_params['n_epochs']):
        print("Epoch %d/%d" % (epoch,self.training_params['n_epochs']))
        print("Elapsed time : %.0f seconds" % (time.time() - start))
        for xx_pad, yy_pad_neg, x_lens in tqdm.tqdm(train_dataloader):
          self.train()
          optimizer.zero_grad()
          loss = self.compute_loss_batch(xx_pad.to(get_device()), yy_pad_neg.to(get_device()))
          loss.backward()
          if self.training_params['clip'] :
            clip_grad_norm_(self.parameters(), max_norm=self.training_params['clip'], norm_type=2)
          optimizer.step()
          if print_every :
            if batch_ct % self.training_params["step_every"] == 0:
              scheduler.step()
              print(loss.item())
              recos = self.compute_recos(test_dataloader)
              r_prec = test_evaluator.compute_all_R_precisions(recos)
              ndcg = test_evaluator.compute_all_ndcgs(recos)
              click = test_evaluator.compute_all_clicks(recos)
              print("rprec : %.3f, ndcg : %.3f, click : %.3f" % (r_prec.mean(), ndcg.mean(), click.mean()))
          batch_ct += 1
        if savePath:
          torch.save(self, savePath)
      return

    def beam_search(self, X_seed, all_rep, n_recos, beam_size, lv=- 10 ** 3, debug = False):
        # Computes Beam search for inference instead of top-K (was not used for results reported in the article)
        all_top_k = []
        all_log_probs = []
        dev = get_device()
        X_seed = X_seed.long().to(dev)

        n_seed = X_seed.shape[1]
        N = all_rep.shape[0]
        X_seed = X_seed.to(dev)
        bs = X_seed.shape[0]
        if debug :
            scores_dict = [{"log_proba" : 1.0} for i in range(bs)]
        X_rep = self.representer(X_seed)
        mask = torch.zeros_like(X_seed).to(dev)
        X_agg = self.aggregator.aggregate_single(X_rep, mask)
        cross_product = torch.matmul(X_agg, all_rep.T)
        cross_product = torch.log_softmax(cross_product, dim=1)
        cross_product = cross_product.scatter(1, X_seed.long(), value=lv).cpu()
        cross_product[:, 0] = lv
        cross_product[:, 1] = lv
        top_k = cross_product.topk(beam_size, dim=1)
        current_log_probas = top_k[0].to(dev)
        first_preds = top_k[1].reshape(-1).unsqueeze(1)
        X_beam = torch.cat([X_seed.unsqueeze(2).repeat(1, 1, beam_size).reshape((beam_size * bs, -1)), first_preds.to(dev)],
                           dim=1)
        if debug :
            all_top_k.append(X_beam)
            all_log_probs.append(current_log_probas)
        l = X_beam.shape[1]
        while l < n_seed + n_recos:
          X_rep = self.representer(X_beam)
          mask = torch.zeros_like(X_beam).to(dev)
          X_agg = self.aggregator.aggregate_single(X_rep, mask)
          cross_product = torch.matmul(X_agg, all_rep.T)
          cross_product = torch.log_softmax(cross_product, dim=1)
          cross_product = cross_product.scatter(1, X_beam.long(), value=lv)
          cross_product[:, 0] = lv
          cross_product[:, -1] = lv
          log_probs = current_log_probas.unsqueeze(2).repeat((1, 1, N)).reshape((bs, beam_size * N))
          cross_product = cross_product.reshape((bs, beam_size * N))
          log_probs = log_probs + cross_product
          top_k = log_probs.topk(beam_size, dim=1)
          track_ids = torch.remainder(top_k[1], N)
          current_log_probas = top_k[0]
          seqs = torch.div(top_k[1], N, rounding_mode="floor")
          X_beam = torch.cat(
            [X_beam.reshape((bs, beam_size, -1)).gather(dim=1, index=seqs.unsqueeze(2).repeat((1, 1, l))).reshape((-1, l)),
             track_ids.reshape(-1).unsqueeze(1)], dim=1)
          if debug:
              all_top_k.append(X_beam)
              all_log_probs.append(current_log_probas)
          l = X_beam.shape[1]
        hp_idx = beam_size * torch.arange(bs)
        if debug:
            return X_beam[hp_idx, n_seed:] - 1, all_top_k, all_log_probs
        return X_beam[hp_idx, n_seed:] - 1


    def beam_search_recos(self, test_dataloader, n_recos=10, beam_size=3):
        # Iterate over test set with beam search recommendations
      dev = get_device()
      n_p = len(test_dataloader.dataset)
      with torch.no_grad():
        self.eval()
        recos = Tensor()
        all_rep = self.representer.compute_all_representations()
        for X in tqdm.tqdm(test_dataloader):
          X = X.long().to(dev)
          X_beam = self.beam_search(X, all_rep, n_recos, beam_size)
          recos = torch.cat([recos, (X_beam).cpu().detach()])
      return recos.numpy()