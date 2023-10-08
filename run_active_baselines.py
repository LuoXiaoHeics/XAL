# Concat in Train Dev and Test

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import gc
import os
from numpy.core.fromnumeric import argsort
import torch
import logging
import random
import numpy as np
# from torch._C import half
from torch.distributions.categorical import Categorical

from model import *
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME,RobertaConfig,RobertaTokenizer,T5Tokenizer
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.nn import Softmax
# from model import *
import glob
import json
import shutil
import re
from glue_utils import *
from scipy.stats import entropy

from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors




logger = logging.getLogger(__name__)
try:
    from apex import amp
except ImportError:
    amp = None
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())
ALL_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
    'bert-base-german-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    'bert-large-cased-whole-word-masking-finetuned-squad',
    'bert-base-cased-finetuned-mrpc',
    'bert-base-german-dbmdz-cased',
    'bert-base-german-dbmdz-uncased',
    'xlnet-base-cased',
    'xlnet-large-cased'
)



MODEL_CLASSES = {
    # 'roberta' : (RobertaConfig, RobertaStance, RobertaBase, RobertaTokenizer),
    'flan_t5' : (T5Config, T5ForStanceGeneration, T5Tokenizer),
}

Processor_CLASSES = {
    "stance":SentProcessorStance, 
    "scidoc":SentProcessorSci,
    "rte":SentProcessorRTE, 
    "mrpc":SentProcessorMRPC,
    "deba":SentProcessorDEBA,
    'mams':SentProcessorMAMS
}

label_num = {
    "stance":3, 
    "scidoc":2,
    "rte":2, 
    "mrpc":2,
    "deba":3,
    'mams':3
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument("--data_dir", default='data/fdu-mtl/', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='roberta', type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--fix_tfm", default=0, type=int, help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--target", default=0, type=int,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--do_rational", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--case_study", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--do_dev", action='store_true',
                        help="Whether to run dev.")

    parser.add_argument("--task", default="stance", type=str, choices=["covid19", "clef", "rte", "mrpc", "deba",'mams'])
    parser.add_argument("--reason_seq_length", default=60, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--initial', type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--l_g", default=0.1, type=float,
    #                     help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "constant", "inv_sqrt"])
    parser.add_argument("--optimizer", default="adam", type=str, choices=["adam", "adafactor"])
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed learning")
    parser.add_argument("--lambda", type=float, default=1.0,
                        help="Distributed learning")
    parser.add_argument('--fp16', default=False, action="store_true")
    
    parser.add_argument("--method", default="max_entropy", type=str, choices=["max_entropy", "BALD", "random",'LC','BK','CA','coreset'])


    args = parser.parse_args()
    if args.fp16 and amp is None:
        print("No apex installed, fp16 not used.")
        args.fp16 = False
    return args


def train(args, train_dataset,tokenizer, model,iter_id = 0):
    """ Train the model """
    tb_writer = SummaryWriter(args.output_dir+'_iter_'+str(iter_id))
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params'      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,
                              relative_step=False)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
    elif args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    else:
        scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # For reproducibility (even between python 2 and 3)
    should_stop = False
    best_p = 0.0
    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader,desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            loss = torch.tensor(0, dtype=float).to(args.device)

            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2]}
            l1,logits = model(**inputs)
            l = l1
            if args.n_gpu >1:
                loss += l.mean()
            else :
                loss += l

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0 or global_step == 1:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("epoch: {:d}, step: {:d}, "
                                "loss: {:.4f}, lr: {:g}".format(epoch, global_step,
                                                                (tr_loss - logging_loss) / args.logging_steps,  
                                                                scheduler.get_lr()[0]))
                    logging_loss = tr_loss

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer, mode="val")
                    logger.info("macro-f1 {:4f}".format(results['macro-f1']))
                    tb_writer.add_scalar('dev_best_f1', global_step)
                    if best_p < results['macro-f1']:
                        best_p = results['macro-f1']
                        if not os.path.exists(args.output_dir+'_iter_'+str(iter_id)):
                            os.mkdir(args.output_dir+'_iter_'+str(iter_id))
                        model.save_pretrained(args.output_dir+'_iter_'+str(iter_id))
                        tokenizer.save_pretrained(args.output_dir+'_iter_'+str(iter_id))
                        torch.save(args, os.path.join(args.output_dir+'_iter_'+str(iter_id), 'training_args.bin'))
                        logger.info("Saving best model checkpoint.")

                    tb_writer.add_scalar('eval_best_p', best_p, global_step)

                if 0 < args.max_steps < global_step:
                    should_stop = True
        if should_stop:
            break

    del train_dataloader
    tb_writer.close()

def load_and_cache_examples(args, tokenizer, mode='train', ids = None):
    processor = Processor_CLASSES[args.task]()
    data_dir = args.data_dir+'/'+args.task
    # logger.info("Creating features from dataset file at %s", args.data_dir)
    if mode == 'train':
        examples = processor.get_train_examples(data_dir, ids = ids,using_score = False)
        features_w_r = convert_exp_reas_to_features_modify(examples=examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer,input_=False,max_reason = args.reason_seq_length)
        labels_ids = torch.tensor([f.input_ids for f in features_w_r], dtype=torch.long)
        labels_ids[labels_ids == tokenizer.pad_token_id] = -100

    elif mode == 'test':
        examples = processor.get_test_examples(data_dir,using_score = False)
    elif mode == 'val':
        examples = processor.get_dev_examples(data_dir,using_score = False)
    else:
        raise Exception("Invalid data mode %s..." % mode)

    features_wo_r = convert_exp_reas_to_features_modify(examples=examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer,input_=True)
    
    all_input_ids = torch.tensor([f.input_ids for f in features_wo_r], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features_wo_r], dtype=torch.long)
    cls_label_ids = torch.tensor([f.label for f in features_wo_r], dtype=torch.long)   
    if mode == 'train':
        ids = torch.tensor([f.id for f in features_wo_r], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, cls_label_ids, labels_ids,ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, cls_label_ids)
    return dataset

def compute_metrics_absa(preds, labels):
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    scores = {'macro-f1': macro_f1}
    print(scores)
    return scores

def evaluate(args, model, tokenizer, mode):
    eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    ids = []
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2],                  
                        }
            ids.extend(batch[0])
            ls,logits = model(**inputs)
            logit_list.append(torch.argmax(logits, axis=-1))
            label_list.append(batch[2])
        eval_steps += 1
    preds = torch.cat(logit_list, axis=0)
    labels= torch.cat(label_list, axis=0)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    result = compute_metrics_absa(preds, labels)
    return result

def random_select(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    indices = [i for i in range(len(train_dataset))]

    random.shuffle(indices)
    save_ids = []
    save_num = 0

    for idc in indices:
        if idc in pre_ids:
            continue
        save_ids.append(int(idc))
        save_num+=1
        if save_num >=100:
            break
    save_ids.extend(pre_ids)
    return save_ids 

def least_confidence_select(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    max_conf, label_list = [],[]
    ids = []
    model.eval()
    scores = []
    epoch_iterator = tqdm(eval_dataloader,desc="Iteration")
    
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():   
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2],                  
                        }
            ids.extend(batch[0])
            ls,logits = model(**inputs)
            conf,ind = torch.max(logits, axis=-1)
            max_conf.append(conf)
            label_list.append(batch[2])

    scores = 1 - torch.cat(max_conf, axis=0)
    # ids = torch.cat(ids, axis=0)
    conf, indices = torch.sort(scores,descending=True)
    save_ids = []
    save_num = 0

    for idc in indices:
        if idc in pre_ids:
            continue
        save_ids.append(int(idc.detach().cpu()))
        save_num+=1
        if save_num >=100:
            break
    save_ids.extend(pre_ids)
    return save_ids 

def _best_versus_second_best(proba):
    # print(proba)
    ind = np.argsort(proba)
    return proba[ind[-1]] - proba[ind[-2]]

def breaking_ties(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    max_conf, label_list = [],[]
    ids = []
    model.eval()
    scores = []
    epoch_iterator = tqdm(eval_dataloader,desc="Iteration")
    

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():   
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2],                  
                        }
            ids.extend(batch[0])
            ls,logits = model(**inputs)
            max_conf.append(logits)
            label_list.append(batch[2])


    logits = torch.cat(max_conf, axis=0).cpu().numpy()
    # print(logits.shape)
    scores = np.apply_along_axis(_best_versus_second_best, 1, logits)
    indices = np.argsort(scores)
    save_ids = []
    save_num = 0

    for idc in indices:
        if idc in pre_ids:
            continue
        save_ids.append(int(idc))
        save_num+=1
        if save_num >=100:
            break
    save_ids.extend(pre_ids)
    return save_ids  

def bold_select(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=32)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    ids = []

    model.train()
    def mc_step(batch, k=10):
        bsz = batch[2].shape[0]
        probs, disagreements = [], []
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2],                  
                        }
            for _ in range(k):
                ls,logits = model(**inputs)
                prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                prob = prob.reshape(-1, prob.shape[-1])  # [16, 21, 3] -> [16*21, 3]
                probs.append(prob)
                disagreements.append(entropy(prob.transpose(1, 0)))

        entropies = entropy(np.mean(probs, axis=0).transpose(1, 0))
        disagreements = np.mean(disagreements, axis=0)
        diff = entropies - disagreements
        return diff.reshape(bsz, -1)
        
    scores = []
    epoch_iterator = tqdm(eval_dataloader,desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        info = mc_step(batch, k=10)
        scores.append(info)

    scores = (np.concatenate(scores,axis=0)).reshape(-1)
    indices = np.argsort(scores,axis = -1)

    save_ids = []
    save_num = 0
    for idc in indices:
        if idc in pre_ids:
            continue
        save_ids.append(int(idc))
        save_num+=1
        if save_num >=100:
            break
    save_ids.extend(pre_ids)
    return save_ids
    
def max_entropy_select(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    ids = []
    model.eval()
    scores = []
    epoch_iterator = tqdm(eval_dataloader,desc="Iteration")
    
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():   
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2],                  
                        }
            ids.extend(batch[0])
            ls,logits = model(**inputs)
            logit_list.append(torch.argmax(logits, axis=-1))
            label_list.append(batch[2])

            categorical = Categorical(logits=logits)
            entropies = categorical.entropy()
            scores.append(entropies)  

    scores = torch.cat(scores, axis=0)
    # ids = torch.cat(ids, axis=0)
    conf, indices = torch.sort(scores,descending=True)
    save_ids = []
    save_num = 0
    
    # print(conf)

    for idc in indices:
        if idc in pre_ids:
            continue
        save_ids.append(int(idc.detach().cpu()))
        save_num+=1
        if save_num >=100:
            break
    save_ids.extend(pre_ids)
    return save_ids

def _cosine_distance(a, b, normalized=False):
    sim = np.matmul(a, b.T)
    if not normalized:
        sim = sim / np.dot(np.linalg.norm(a, axis=1)[:, np.newaxis],
                           np.linalg.norm(b, axis=1)[np.newaxis, :])
    return np.arccos(sim) / np.pi

from sklearn.metrics import pairwise_distances

def _euclidean_distance(a, b, normalized=False):
    _ = normalized
    return pairwise_distances(a, b, metric='euclidean')

def _check_coreset_size(x, n):
    if n > x.shape[0]:
        raise ValueError(f'n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})')


def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=8, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(x.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)

    return np.array(ind_new)


def coreset_select(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    ids = []
    representation = []
    model.eval()
    scores = []
    epoch_iterator = tqdm(eval_dataloader,desc="Iteration")

    indices = [i for i in range(len(train_dataset))]
    unlabeled = list(set(indices) - set(pre_ids))

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():   
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2],  
                          'r_repre': True,                
                        }
            ids.extend(batch[0])
            logits,rep = model(**inputs)
            logit_list.append(logits)
            representation.append(rep)
  
    
    rep = torch.cat(representation,axis=0).detach().cpu().numpy()

    new_indices = greedy_coreset(rep,np.array(unlabeled),np.array(pre_ids), 100, distance_metric = 'euclidean')

    save_ids = []
    for u in new_indices:
        save_ids.append(u)
    save_ids.extend(pre_ids)
    return save_ids

from scipy.special import rel_entr
from sklearn.preprocessing import normalize


def _contrastive_active_learning(dataset, embeddings, embeddings_proba,
                                    indices_unlabeled, nn, n):
    scores = []

    embeddings_unlabelled_proba = embeddings_proba[indices_unlabeled]
    embeddings_unlabeled = embeddings[indices_unlabeled]

    num_batches = int(np.ceil(len(dataset) / 4))
    offset = 0
    for batch_idx in np.array_split(np.arange(indices_unlabeled.shape[0]), num_batches,
                                    axis=0):
        print(batch_idx)
        nn_indices = nn.kneighbors(embeddings_unlabeled[batch_idx],
                                    n_neighbors=8,
                                    return_distance=False)
        print(nn_indices)
        kl_divs = np.apply_along_axis(lambda v: np.mean([
            rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
            for i in nn_indices[v - offset]]),
            0,
            batch_idx[None, :])
        print(kl_divs.tolist())
        scores.extend(kl_divs.tolist())
        offset += batch_idx.shape[0]

    scores = np.array(scores)
    indices = np.argpartition(-scores, n)[:n]

    return indices

def CA(args,model,tokenizer,pre_ids):
    train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = None)
    indices = [i for i in range(len(train_dataset))]
    eval_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_loss, eval_steps = 0.0, 0
    logit_list, label_list = [],[]
    representation = []
    ids = []
    model.eval()
    scores = []
    softmax = Softmax(dim=1)
    epoch_iterator = tqdm(eval_dataloader,desc="Iteration")
    
    unlabeled = list(set(indices) - set(pre_ids))

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():   
            
            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'cls_label': batch[2], 
                          'r_repre': True       
                        }

            ids.extend(batch[0])
            logits,rep = model(**inputs)
            logit_list.append(logits)
            representation.append(rep)
    
    logits =  softmax(torch.cat(logit_list,axis=0))
    print(logits)
    logits = logits.detach().cpu().numpy()
    rep = torch.cat(representation,axis=0).detach().cpu().numpy()

    print(logits.shape, rep.shape)

    embeddings = normalize(rep, axis=1)

    nn = NearestNeighbors(n_neighbors=8)
    nn.fit(embeddings)

    new_indices = _contrastive_active_learning(train_dataset, embeddings, logits,
                                            np.array(unlabeled), nn, 100)
    save_ids = []
    for u in new_indices:
        save_ids.append(u)
    save_ids.extend(pre_ids)
    return save_ids



def main():
    args = init_args()
    args.output_dir += str(args.method)+'_'+str(args.task)+'_'+args.model_type+'_initial_' + str(args.initial) +'_s_'+str(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir += '/'

    torch.cuda.set_device(args.gpu_id)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu =1
    
    print("GPU number is : ~~~~~~~~~~~~~~~~  "+ str(args.n_gpu))
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(args)

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          output_hidden_states=True)

    config.sent_number = label_num[args.task]
    config.r = args.do_rational
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir='./cache')

    num_added_toks = tokenizer.add_tokens(['<cls>'])

    if args.do_train:
        for iter_id in range(5):
            if os.path.exists(args.output_dir+'_iter_'+str(iter_id)):
                if os.path.exists(args.output_dir+'_iter_'+str(iter_id)+'/ids.txt'):
                    continue
                
            if iter_id == 0:
                with open(args.data_dir +args.task +'/initial_'+str(args.initial)+'.txt','r') as f:
                    txt = f.readline().split(',')
                    ids = [int(txt[i]) for i in range(len(txt)-1)]
                model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                    config=config, cache_dir='./cache')     
            else:
                with open(args.output_dir+'_iter_'+str(iter_id-1)+'/ids.txt','r') as f:
                    txt = f.readline().split(',')
                    ids = [int(txt[i]) for i in range(len(txt)-1)]
                model = model_class.from_pretrained(args.output_dir+'_iter_'+str(iter_id-1))

            model.resize_token_embeddings(len(tokenizer))
            model.to(args.device)

            if args.n_gpu >1:
                model = torch.nn.DataParallel(model)
            train_dataset = load_and_cache_examples(args, tokenizer, mode='train',ids = ids)
            train(args, train_dataset,tokenizer, model,iter_id =iter_id)

            del train_dataset 
            gc.collect()
            del model
            torch.cuda.empty_cache()

            print('######## Testing')
            args.model_type = args.model_type.lower()
            r = 0
            with open (args.output_dir+'_iter_'+str(iter_id)+'/test_results.txt','w') as f:
                model = model_class.from_pretrained(args.output_dir+'_iter_'+str(iter_id))
                model.to(args.device)
                results = evaluate(args, model, tokenizer, 'test')
                f.write("results "+str(results)+'\n')

            if iter_id == 4:
                break

            print('######## Selecting Saving Samples')
            if args.method == 'max_entropy':
                save_ids = max_entropy_select(args,model,tokenizer,ids)
            elif args.method == 'random':
                save_ids = random_select(args,model,tokenizer,ids)
            elif args.method == 'BALD':
                save_ids = bold_select(args,model,tokenizer,ids)
            elif args.method == 'LC':
                save_ids = least_confidence_select(args,model,tokenizer,ids)
            elif args.method == 'BK':
                save_ids = breaking_ties(args,model,tokenizer,ids)
            elif args.method == 'CA':
                save_ids = CA(args,model,tokenizer,ids)
            elif args.method == 'coreset':
                save_ids = coreset_select(args,model,tokenizer,ids)
            with open(args.output_dir+'_iter_'+str(iter_id)+'/ids.txt','w') as f:
                for u in save_ids:
                    f.write(str(u)+', ')
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
