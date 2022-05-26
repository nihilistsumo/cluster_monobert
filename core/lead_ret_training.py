import argparse
import random

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
from util.data import get_page_sec_para_dict
from core.sec_ret_training import Mono_SBERT_Clustering_Reg_Model, bm25_ranking, random_ranking, prepare_data, is_fit_for_training
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def eval_mono_bert_ranking_full(page_sec_paras, paratext, mode='bm25', model=None, per_query=False):
    if model is not None:
        model.eval()
    with open('temp.qrels', 'w') as f:
        for p in page_sec_paras.keys():
            for s in page_sec_paras[p].keys():
                if '/' not in s:
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
            cand_set_texts = [paratext[p] for p in cand_set]
            for sec in page_sec_paras[page].keys():
                if '/' not in sec:
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


def train_mono_sbert(train_qrels,
                     train_paratext_tsv,
                     b1train_qrels,
                     b1train_paratext_tsv,
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
    page_sec_paras, train_paratext = prepare_data(train_qrels, train_paratext_tsv)
    val_pages = random.sample(page_sec_paras.keys(), val_size)
    val_page_sec_paras = {k:page_sec_paras[k] for k in val_pages}
    train_page_sec_paras = {k:page_sec_paras[k] for k in page_sec_paras.keys() - val_pages}
    b1train_page_sec_paras, b1train_paratext = prepare_data(b1train_qrels, b1train_paratext_tsv)
    b1test_page_sec_paras, b1test_paratext = prepare_data(b1test_qrels, b1test_paratext_tsv)

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
                    if '/' not in sec:
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
    parser.add_argument('-trq', '--train_qrels', default='train/base.train.cbor-without-by1-toplevel.qrels')
    parser.add_argument('-trp', '--train_ptext', default='train/train_paratext.tsv')
    parser.add_argument('-tq1', '--t1_qrels', default='benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels')
    parser.add_argument('-tp1', '--t1_ptext', default='benchmarkY1-train-nodup/by1train_paratext/by1train_paratext.tsv')
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

    train_mono_sbert(args.dataset_dir + '/' + args.train_qrels,
                         args.dataset_dir + '/' + args.train_ptext,
                         args.dataset_dir + '/' + args.t1_qrels,
                         args.dataset_dir + '/' + args.t1_ptext,
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