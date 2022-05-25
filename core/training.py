import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import transformers
from transformers import AdamW
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import models, SentenceTransformer
import ir_measures
from ir_measures import MAP, Rprec, nDCG
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from util.data import get_article_qrels, get_page_sec_para_dict, TRECCAR_Datset
from core.clustering import get_nmi_loss, get_weighted_adj_rand_loss, get_adj_rand_loss, get_rand_loss
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


def prepare_data(art_qrels, qrels, paratext_tsv):
    page_sec_paras = get_page_sec_para_dict(qrels)
    paratext = {}
    with open(paratext_tsv, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            paratext[p] = l.split('\t')[1].strip()
    return page_sec_paras, paratext


def eval_clustering(test_samples, model):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_samples:
        true_labels = s.para_labels
        k = len(set(true_labels))
        texts = s.para_texts
        query_content = s.q.split('enwiki:')[1].replace('%20', ' ')
        input_texts = [(query_content, t, '') for t in texts]
        embeddings = model.emb_model.encode(input_texts, convert_to_tensor=True)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def random_ranking(cand_set):
    n = len(cand_set)
    return torch.rand(n).tolist()


def bm25_ranking(sec, cand_set_texts):
    if '/' in sec:
        query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
    else:
        query = sec.replace('enwiki:', '').replace('%20', ' ')
    tokenized_query = query.split(' ')
    tokenized_corpus = [t.split(' ') for t in cand_set_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(tokenized_query)


def eval_mono_bert_bin_clustering_full(model, page_paras, page_sec_paras, paratext, qrels):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            n = len(cand_set)
            for sec in page_sec_paras[page].keys():
                cand_set_texts = [paratext[p] for p in cand_set]
                pred_score = model.get_rank_scores(sec, cand_set_texts)
                for i in range(n):
                    f.write(sec + ' 0 ' + cand_set[i] + ' 0 ' + str(pred_score[i]) + ' val_runid\n')
                binary_cluster_labels = [1 if cand_set[i] in page_sec_paras[page][sec] else 0 for i in range(n)]
                pred_cluster_labels = model.get_binary_clustering(sec, cand_set_texts)
                rand = adjusted_rand_score(binary_cluster_labels, pred_cluster_labels)
                nmi = normalized_mutual_info_score(binary_cluster_labels, pred_cluster_labels)
                rand_dict[page + sec] = rand
                nmi_dict[page + sec] = nmi
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals, rand_dict, nmi_dict


def eval_mono_bert_bin_clustering(model, samples):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in range(len(samples)):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            sections = list(set(s.para_labels))
            for sec in sections:
                pred_score = model.get_rank_scores(sec, s.para_texts)
                for p in range(n):
                    f.write(sec + ' 0 ' + cand_set[p] + ' 0 ' + str(pred_score[p]) + ' val_runid\n')
                binary_cluster_labels = [1 if sec == s.para_labels[i] else 0 for i in range(n)]
                pred_cluster_labels = model.get_binary_clustering(sec, s.para_texts)
                rand = adjusted_rand_score(binary_cluster_labels, pred_cluster_labels)
                nmi = normalized_mutual_info_score(binary_cluster_labels, pred_cluster_labels)
                rand_dict[s.q + sec] = rand
                nmi_dict[s.q + sec] = nmi
    qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
    run_dat = ir_measures.read_trec_run('temp.val.run')
    rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals, rand_dict, nmi_dict


def eval_mono_bert_ranking(model, samples):
    model.eval()
    with open('temp.val.qrels', 'w') as f:
        for s in samples:
            for i in range(len(s.paras)):
                f.write(s.para_labels[i] + ' 0 ' + s.paras[i] + ' 1\n')
    with open('temp.val.run', 'w') as f:
        for si in range(len(samples)):
            s = samples[si]
            cand_set = s.paras
            n = len(cand_set)
            sections = list(set(s.para_labels))
            for sec in sections:
                pred_score = model.get_rank_scores(sec, s.para_texts)
                for p in range(n):
                    f.write(sec + ' 0 ' + cand_set[p] + ' 0 ' + str(pred_score[p]) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels('temp.val.qrels')
    run_dat = ir_measures.read_trec_run('temp.val.run')
    rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals


def eval_mono_bert_ranking_full(page_sec_paras, paratext, mode='bm25', model=None, per_query=False):
    if model is not None:
        model.eval()
    with open('temp.qrels', 'w') as f:
        for p in page_sec_paras.keys():
            for s in page_sec_paras[p].keys():
                for para in page_sec_paras[p][s]:
                    f.write(s + ' 0 ' + para + ' 1\n')
    with open('temp.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = []
            for s in page_sec_paras[page].keys():
                cand_set += page_sec_paras[page][s]
            n = len(cand_set)
            for sec in page_sec_paras[page].keys():
                cand_set_texts = [paratext[p] for p in cand_set]
                if mode == 'bm25':
                    pred_score = bm25_ranking(sec, cand_set_texts)
                elif mode == 'rand':
                    pred_score = random_ranking(cand_set)
                elif mode == 'model' and model is not None:
                    pred_score = model.get_rank_scores(sec, cand_set_texts)
                else:
                    print(mode + ' not recognized or model is None')
                    return None
                for i in range(n):
                    f.write(sec + ' 0 ' + cand_set[i] + ' 0 ' + str(pred_score[i]) + ' runid\n')
    qrels_dat = ir_measures.read_trec_qrels('temp.qrels')
    run_dat = ir_measures.read_trec_run('temp.run')
    if per_query:
        rank_evals = ir_measures.iter_calc([MAP, Rprec, nDCG], qrels_dat, run_dat)
    else:
        rank_evals = ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)
    return rank_evals


def is_fit_for_training(paras, sec_paras):
    return 10 <= len(paras) <= 200 and len(sec_paras.keys()) > 2


class Mono_SBERT_Clustering_Reg_Model(nn.Module):
    def __init__(self, sbert_emb_model, device, kmeans_plus=False):
        super(Mono_SBERT_Clustering_Reg_Model, self).__init__()
        self.emb_model = sbert_emb_model
        self.fc1 = nn.Linear(in_features=sbert_emb_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()
        self.device = device

    def forward(self, sec, para_texts):
        input_sec_para_texts = []
        if '/' in sec:
            query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
        else:
            query = sec.replace('enwiki:', '').replace('%20', ' ')
        for pi in range(len(para_texts)):
            input_sec_para_texts.append((query, para_texts[pi]))
        input_fet = self.emb_model.tokenize(input_sec_para_texts)
        put_features_in_device(input_fet, self.device)
        output_emb = self.emb_model(input_fet)['sentence_embedding']
        self.pred_score = self.act(self.fc1(output_emb)).flatten()
        dist_mat = torch.cdist(output_emb, output_emb)
        self.sim_mat = 2 / (1 + torch.exp(dist_mat))
        return self.pred_score, self.sim_mat

    def get_qs_embeddings(self, sec, para_texts):
        input_sec_para_texts = []
        if '/' in sec:
            query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
        else:
            query = sec.replace('enwiki:', '').replace('%20', ' ')
        for pi in range(len(para_texts)):
            input_sec_para_texts.append((query, para_texts[pi]))
        output_emb = self.emb_model.encode(input_sec_para_texts, convert_to_tensor=True)
        return output_emb

    def get_rank_scores(self, sec, para_texts):
        output_emb = self.get_qs_embeddings(sec, para_texts).to(self.device)
        pred_score = self.act(self.fc1(output_emb)).flatten().detach().cpu().numpy()
        return pred_score


def train_mono_sbert_with_clustering_reg(treccar_data,
                                            val_art_qrels,
                                            val_qrels,
                                            val_paratext_tsv,
                                            test_art_qrels,
                                            test_qrels,
                                            test_paratext_tsv,
                                            device,
                                            model_out,
                                            trans_model_name,
                                            max_len,
                                            max_grad_norm,
                                            weight_decay,
                                            warmup,
                                            lrate,
                                            num_epochs,
                                            val_step,
                                            lambda_val):
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    dataset = np.load(treccar_data, allow_pickle=True)[()]['data']
    train_samples = dataset.samples
    val_samples = dataset.val_samples
    test_samples = dataset.test_samples
    #### Smaller experiment ####
    train_samples = train_samples[:10000]
    ############################
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    model = Mono_SBERT_Clustering_Reg_Model(emb_model, device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_samples)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    mse = nn.MSELoss()
    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nInitial val MAP: %.4f' % val_rank_eval[MAP])
    rand_rank_eval, rand_rand_dict, rand_nmi_dict = eval_random_ranker(val_samples)
    print('\nRandom ranker performance val MAP: %.4f' % rand_rank_eval[MAP])
    val_eval_score = val_rank_eval[MAP]
    bm25_rank_eval = eval_bm25_ranker(val_samples)
    print('\nBM25 ranker performance val MAP: %.4f' % bm25_rank_eval[MAP])
    for epoch in range(num_epochs):
        print('Epoch %3d' % (epoch + 1))
        for i in tqdm(range(train_data_len)):
            sample = train_samples[i]
            k = len(set(sample.para_labels))
            '''
            if k > 4:
                continue
            '''
            n = len(sample.paras)
            model.train()
            true_sim_mat = torch.zeros((n, n)).to(device)
            for p in range(n):
                for q in range(n):
                    if sample.para_labels[p] == sample.para_labels[q]:
                        true_sim_mat[p][q] = 1.0
            for sec in set(sample.para_labels):
                pred_score, sim_mat = model(sec, sample.para_texts)
                true_labels = [1.0 if sec == sample.para_labels[p] else 0 for p in range(len(sample.para_labels))]
                true_labels_tensor = torch.tensor(true_labels).to(device)
                rk_loss = mse(pred_score, true_labels_tensor)
                cl_loss = mse(sim_mat, true_sim_mat)
                loss = lambda_val * rk_loss + (1 - lambda_val) * cl_loss
                loss.backward()
                # print('Rank loss: %.4f, Cluster loss: %.4f, Loss: %.4f' % (rk_loss.item(), cl_loss.item(), loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if (i + 1) % val_step == 0:
                val_rank_eval = eval_mono_bert_ranking(model, val_samples)
                print('\nval MAP: %.4f' % val_rank_eval[MAP])
                if val_rank_eval[MAP] > val_eval_score and model_out is not None:
                    torch.save(model, model_out)
                    val_eval_score = val_rank_eval[MAP]

    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nval MAP: %.4f' % val_rank_eval[MAP])
    if val_rank_eval[MAP] > val_eval_score and model_out is not None:
        torch.save(model, model_out)
    print('\nTraining complete. Evaluating on full val and test sets...')
    val_rank_eval = eval_mono_bert_ranking_full(model, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
    print('\nFull val eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        val_rank_eval[MAP], val_rank_eval[Rprec], val_rank_eval[nDCG]))
    test_rank_eval = eval_mono_bert_ranking_full(model, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nFull test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        test_rank_eval[MAP], test_rank_eval[Rprec], test_rank_eval[nDCG]))


def train_mono_sbert_with_bin_clustering_reg(treccar_data,
                                            val_art_qrels,
                                            val_qrels,
                                            val_paratext_tsv,
                                            test_art_qrels,
                                            test_qrels,
                                            test_paratext_tsv,
                                            device,
                                            model_out,
                                            trans_model_name,
                                            max_len,
                                            max_grad_norm,
                                            weight_decay,
                                            warmup,
                                            lrate,
                                            num_epochs,
                                            val_step,
                                            lambda_val):
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    dataset = np.load(treccar_data, allow_pickle=True)[()]['data']
    train_samples = dataset.samples
    val_samples = dataset.val_samples
    test_samples = dataset.test_samples
    #### Smaller experiment ####
    train_samples = train_samples[:10000]
    ############################
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    model = Mono_SBERT_Clustering_Reg_Model(emb_model, device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_samples)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    mse = nn.MSELoss()
    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nInitial val MAP: %.4f' % val_rank_eval[MAP])
    rand_rank_eval, rand_rand_dict, rand_nmi_dict = eval_random_ranker(val_samples)
    print('\nRandom ranker performance val MAP: %.4f' % rand_rank_eval[MAP])
    val_eval_score = val_rank_eval[MAP]
    bm25_rank_eval = eval_bm25_ranker(val_samples)
    print('\nBM25 ranker performance val MAP: %.4f' % bm25_rank_eval[MAP])
    for epoch in range(num_epochs):
        print('Epoch %3d' % (epoch + 1))
        for i in tqdm(range(train_data_len)):
            sample = train_samples[i]
            k = len(set(sample.para_labels))
            '''
            if k > 4:
                continue
            '''
            n = len(sample.paras)
            model.train()
            for sec in set(sample.para_labels):
                true_sim_mat = torch.zeros((n, n)).to(device)
                for p in range(n):
                    for q in range(n):
                        if sample.para_labels[p] == sample.para_labels[q] == sec:
                            true_sim_mat[p][q] = 1.0
                        elif sample.para_labels[p] != sec and sample.para_labels[q] != sec:
                            true_sim_mat[p][q] = 1.0
                pred_score, sim_mat = model(sec, sample.para_texts)
                true_labels = [1.0 if sec == sample.para_labels[p] else 0 for p in range(len(sample.para_labels))]
                true_labels_tensor = torch.tensor(true_labels).to(device)
                rk_loss = mse(pred_score, true_labels_tensor)
                cl_loss = mse(sim_mat, true_sim_mat)
                loss = lambda_val * rk_loss + (1 - lambda_val) * cl_loss
                loss.backward()
                # print('Rank loss: %.4f, Cluster loss: %.4f, Loss: %.4f' % (rk_loss.item(), cl_loss.item(), loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if (i + 1) % val_step == 0:
                val_rank_eval = eval_mono_bert_ranking(model, val_samples)
                print('\nval MAP: %.4f' % val_rank_eval[MAP])
                if val_rank_eval[MAP] > val_eval_score and model_out is not None:
                    torch.save(model, model_out)
                    val_eval_score = val_rank_eval[MAP]

    val_rank_eval = eval_mono_bert_ranking(model, val_samples)
    print('\nval MAP: %.4f' % val_rank_eval[MAP])
    if val_rank_eval[MAP] > val_eval_score and model_out is not None:
        torch.save(model, model_out)
    print('\nTraining complete. Evaluating on full val and test sets...')
    val_rank_eval = eval_mono_bert_ranking_full(model, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
    print('\nFull val eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        val_rank_eval[MAP], val_rank_eval[Rprec], val_rank_eval[nDCG]))
    test_rank_eval = eval_mono_bert_ranking_full(model, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nFull test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        test_rank_eval[MAP], test_rank_eval[Rprec], test_rank_eval[nDCG]))


def train_mono_sbert(train_art_qrels,
                     train_qrels,
                     train_paratext_tsv,
                     b1train_art_qrels,
                     b1train_qrels,
                     b1train_paratext_tsv,
                     b1test_art_qrels,
                     b1test_qrels,
                     b1test_paratext_tsv,
                     device,
                     model_out,
                     trans_model_name,
                     max_len,
                     max_grad_norm,
                     weight_decay,
                     warmup,
                     lrate,
                     num_epochs,
                     val_step,
                     lambda_val,
                     val_size,
                     bin_cluster_mode):
    page_sec_paras, train_paratext = prepare_data(train_art_qrels, train_qrels, train_paratext_tsv)
    val_pages = random.sample(page_sec_paras.keys(), val_size)
    val_page_sec_paras = {k:page_sec_paras[k] for k in val_pages}
    train_page_sec_paras = {k:page_sec_paras[k] for k in page_sec_paras.keys() - val_pages}
    b1train_page_sec_paras, b1train_paratext = prepare_data(b1train_art_qrels, b1train_qrels, b1train_paratext_tsv)
    b1test_page_sec_paras, b1test_paratext = prepare_data(b1test_art_qrels, b1test_qrels, b1test_paratext_tsv)

    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    model = Mono_SBERT_Clustering_Reg_Model(emb_model, device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = len(train_page_sec_paras.keys())
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    mse = nn.MSELoss()
    val_rank_eval = eval_mono_bert_ranking_full(val_page_sec_paras, train_paratext, mode='model', model=model)
    print('\nInitial val MAP: %.4f' % val_rank_eval[MAP])
    rand_rank_eval = eval_mono_bert_ranking_full(val_page_sec_paras, train_paratext, mode='rand')
    print('\nRandom ranker performance val MAP: %.4f' % rand_rank_eval[MAP])
    val_eval_score = val_rank_eval[MAP]
    bm25_rank_eval = eval_mono_bert_ranking_full(val_page_sec_paras, train_paratext, mode='bm25')
    print('\nBM25 ranker performance val MAP: %.4f' % bm25_rank_eval[MAP])
    for epoch in range(num_epochs):
        print('Epoch %3d' % (epoch + 1))
        train_pages = list(train_page_sec_paras.keys())
        for i in tqdm(range(train_data_len)):
            page = train_pages[i]
            paras, para_labels = [], []
            for sec in train_page_sec_paras[page].keys():
                paras += train_page_sec_paras[page][sec]
                para_labels += [sec] * len(train_page_sec_paras[page][sec])
            if is_fit_for_training(paras, page_sec_paras[page]):
                para_texts = [train_paratext[p] for p in paras]
                n = len(paras)
                model.train()
                if not bin_cluster_mode:
                    true_sim_mat = torch.zeros((n, n)).to(device)
                    for p in range(n):
                        for q in range(n):
                            if para_labels[p] == para_labels[q]:
                                true_sim_mat[p][q] = 1.0
                for sec in set(para_labels):
                    if bin_cluster_mode:
                        true_sim_mat = torch.zeros((n, n)).to(device)
                        for p in range(n):
                            for q in range(n):
                                if para_labels[p] == para_labels[q] == sec:
                                    true_sim_mat[p][q] = 1.0
                                elif para_labels[p] != sec and para_labels[q] != sec:
                                    true_sim_mat[p][q] = 1.0
                    pred_score, sim_mat = model(sec, para_texts)
                    true_labels = [1.0 if sec == para_labels[p] else 0 for p in range(len(para_labels))]
                    true_labels_tensor = torch.tensor(true_labels).to(device)
                    rk_loss = mse(pred_score, true_labels_tensor)
                    cl_loss = mse(sim_mat, true_sim_mat)
                    loss = lambda_val * rk_loss + (1 - lambda_val) * cl_loss
                    loss.backward()
                    # print('Rank loss: %.4f, Cluster loss: %.4f, Loss: %.4f' % (rk_loss.item(), cl_loss.item(), loss.item()))
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    schd.step()
            if (i + 1) % val_step == 0:
                val_rank_eval = eval_mono_bert_ranking_full(val_page_sec_paras, train_paratext, mode='model', model=model)
                print('\nval MAP: %.4f' % val_rank_eval[MAP])
                if val_rank_eval[MAP] > val_eval_score and model_out is not None:
                    torch.save(model, model_out)
                    val_eval_score = val_rank_eval[MAP]

    val_rank_eval = eval_mono_bert_ranking_full(val_page_sec_paras, train_paratext, mode='model', model=model)
    print('\nval MAP: %.4f' % val_rank_eval[MAP])
    if val_rank_eval[MAP] > val_eval_score and model_out is not None:
        torch.save(model, model_out)
    print('\nTraining complete. Evaluating on by1train and by1test sets...')
    b1train_rank_eval = eval_mono_bert_ranking_full(b1train_page_sec_paras, b1train_paratext, mode='model', model=model)
    print('\nby1train eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        b1train_rank_eval[MAP], b1train_rank_eval[Rprec], b1train_rank_eval[nDCG]))
    b1test_rank_eval = eval_mono_bert_ranking_full(b1test_page_sec_paras, b1test_paratext, mode='model', model=model)
    print('\nby1test eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (
        b1test_rank_eval[MAP], b1test_rank_eval[Rprec], b1test_rank_eval[nDCG]))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Neural ranking')
    parser.add_argument('-dr', '--dataset_dir', default='/home/sk1105/sumanta/QSC_data')
    parser.add_argument('-tra', '--train_art_qrels', default='train/base.train.cbor-without-by1-article.qrels')
    parser.add_argument('-trq', '--train_qrels', default='train/base.train.cbor-without-by1-toplevel.qrels')
    parser.add_argument('-trp', '--train_ptext', default='train/train_paratext.tsv')
    parser.add_argument('-ta1', '--t1_art_qrels', default='benchmarkY1-train-nodup/train.pages.cbor-article.qrels')
    parser.add_argument('-tq1', '--t1_qrels', default='benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels')
    parser.add_argument('-tp1', '--t1_ptext', default='benchmarkY1-train-nodup/by1train_paratext/by1train_paratext.tsv')
    parser.add_argument('-ta2', '--t2_art_qrels', default='benchmarkY1-test-nodup/test.pages.cbor-article.qrels')
    parser.add_argument('-tq2', '--t2_qrels', default='benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels')
    parser.add_argument('-tp2', '--t2_ptext', default='benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv')
    parser.add_argument('-op', '--output_model', default=None)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-bt', '--batch', type=int, default=8)
    parser.add_argument('-nt', '--max_num_tokens', type=int, help='Max no. of tokens', default=128)
    parser.add_argument('-gn', '--max_grad_norm', type=float, help='Max gradient norm', default=1.0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-wu', '--warmup', type=int, default=10000)
    parser.add_argument('-lr', '--lrate', type=float, default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, default=1)
    parser.add_argument('-ss', '--switch_step', type=int, default=1000)
    parser.add_argument('-ld', '--lambda_val', type=float, default=0.5)
    parser.add_argument('-vs', '--val_size', type=int, default=100)
    parser.add_argument('--bin_cluster', action='store_true', default=False)
    parser.add_argument('--ret', action='store_true', default=False)
    parser.add_argument('--clust', action='store_true', default=False)

    args = parser.parse_args()

    train_mono_sbert(args.dataset_dir + '/' + args.train_art_qrels,
                         args.dataset_dir + '/' + args.train_qrels,
                         args.dataset_dir + '/' + args.train_ptext,
                         args.dataset_dir + '/' + args.t1_art_qrels,
                         args.dataset_dir + '/' + args.t1_qrels,
                         args.dataset_dir + '/' + args.t1_ptext,
                         args.dataset_dir + '/' + args.t2_art_qrels,
                         args.dataset_dir + '/' + args.t2_qrels,
                         args.dataset_dir + '/' + args.t2_ptext,
                         device,
                         args.output_model,
                         args.model_name,
                         args.max_num_tokens,
                         args.max_grad_norm,
                         args.weight_decay,
                         args.warmup,
                         args.lrate,
                         args.epochs,
                         args.switch_step,
                         args.lambda_val,
                         args.val_size,
                         args.bin_cluster)


if __name__ == '__main__':
    main()