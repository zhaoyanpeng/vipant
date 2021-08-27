from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry
from sklearn import metrics

import csv
import math
import copy
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from collections import defaultdict
from clip import _tokenizer, LayerNorm, Transformer, ModifiedResNet, VisualTransformer
from coco_caption.eval_metrics import evaluate_metrics as ac_metric # audio-captioning metric

from .loss_head import build_loss_head, LossHead


class BCELossHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.normalized = False
        assert "output_dim" in kwargs, f"`the label number` is not found in `kwargs`"
        nlabel = kwargs["output_dim"]
        layers = list()
        embed_dim = cfg.embed_dim or cfg.width
        sizes = [embed_dim] + list(cfg.layers) + [nlabel]
        for i in range(len(sizes) - 2):
            layers.extend([
                LayerNorm(sizes[i]),
                nn.Linear(sizes[i], sizes[i + 1]),
            ])
        layers.extend([
            LayerNorm(sizes[-2]),
            nn.Linear(sizes[-2], sizes[-1], bias=cfg.bias)
        ])
        self.linear = nn.Sequential(*layers)
        self.logit_scale = (
            nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if cfg.scaling else
            torch.ones([], requires_grad=False) * np.log(1 / 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.reduce = False 
    
    def copy_state_dict(self, state_dict): 
        key = "logit_scale"
        new_dict = self.state_dict()
        new_dict.update({key: state_dict[key]})
        self.load_state_dict(new_dict)

    def infer(self, x1, x2, *args, **kwargs):
        if not hasattr(self, "audios") or not hasattr(self, "x1s") or \
            not hasattr(self, "x2s") or not hasattr(self, "ids"): 
            self.audios, self.x1s, self.x2s, self.ids = [], [], [], []
        self.audios.append(x1)
        logit_scale = self.logit_scale.exp()
        logits_per_x1 = logit_scale * self.linear(x1)
        loss_mean_x1 = self.loss_fn(logits_per_x1, x2.float())
        predictions = torch.sigmoid(logits_per_x1)
        self.x1s.append(predictions)
        self.x2s.append(x2)
        names = kwargs.get("names", None)
        if names is not None:
            self.ids.extend(names)
        return loss_mean_x1

    def report(self, gold_file=None, **kwargs):
        x1s = torch.cat(self.x1s).cpu().numpy()
        x2s = torch.cat(self.x2s).cpu().numpy()
        nsample, nlabel = x1s.shape[:2]
        
        ap_micro = metrics.average_precision_score(x2s, x1s, average='micro')
        ap_macro = metrics.average_precision_score(x2s, x1s, average='macro')
        ap_weighted = metrics.average_precision_score(x2s, x1s, average='weighted')

        # multi-label classification metrics
        has_err = False
        ap_list, auc_list, precisions, recalls = [], [], [], []
        for k in range(nlabel): # unnecessary (from AST)
            y_true, y_score = x2s[:, k], x1s[:, k]
            ap = metrics.average_precision_score(y_true, y_score, average=None) 
            if math.isnan(ap):
                ap = 0.
                has_err = True
            try:
                auc = metrics.roc_auc_score(y_true, y_score, average=None)
            except Exception as e:
                auc = 0. # auc may not be used a valid metric for this task
                has_err = True
            p, r, _ = metrics.precision_recall_curve(y_true, y_score)
            mid = len(p) // 2
            ap_list.append(ap)
            auc_list.append(auc)
            precisions.append(p[mid])
            recalls.append(r[mid])
        mean_ap = np.mean(ap_list) * 100.
        mean_auc = np.mean(auc_list) * 100.
        mean_p = np.mean(precisions) * 100.
        mean_r = np.mean(recalls) * 100.
        text = (
            f"Err({has_err}) mAP = {mean_ap:2.2f} mAUC = {mean_auc:2.2f} mP = {mean_p:2.2f} mR = {mean_r:2.2f}"
        )

        del self.audios, self.x1s, self.x2s, self.ids
        common = f"Mac-AP = {ap_macro:2.2f} Mic-AP = {ap_micro:2.2f} wAP = {ap_weighted:2.2f}"
        report = f"{common} {text} @ {nsample}" 
        return report

    def forward(self, x1, x2, *args, **kwargs):
        """ x1 is the input features and x2 is the label
        """
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                return self.infer(x1, x2, *args, **kwargs)
            return None 
        logit_scale = self.logit_scale.exp()
        logits_per_x1 = logit_scale * self.linear(x1)
        loss_mean_x1 = self.loss_fn(logits_per_x1, x2.float())
        return loss_mean_x1

class BCEAndCELossHead(LossHead):
    # combining binary cross-entropy loss and cross-entropy loss 
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.normalized = False
        self.loss_ce = build_loss_head(cfg.ce, **kwargs)
        self.loss_bce = build_loss_head(cfg.bce, **kwargs)
        self.lambd_ce = cfg.lambd_ce
        self.reduce = True 

    def report(self, gold_file=None):
        report_ce = ""
        if hasattr(self.loss_ce, "x1s") and hasattr(self.loss_ce, "x2s"):
            report_ce = self.loss_ce.report(gold_file=gold_file)
        report_bce = self.loss_bce.report(gold_file=gold_file) 
        return f"{report_ce}\n{report_bce}" 
    
    def forward(self, x1, x2, *args, x3=None, **kwargs):
        """ x1 is features, x2 is labels, and x3 is mirror features
        """
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                if x3 is not None:
                    self.loss_ce.infer(x1, x3, *args, **kwargs)
                return self.loss_bce.infer(x1, x2, *args, **kwargs)
            return None
        loss_ce = self.loss_ce(x1, x3, *args, **kwargs)
        loss_bce = self.loss_bce(x1, x2, *args, **kwargs)
        loss = self.lambd_ce * loss_ce + loss_bce
        return loss

class ImaginedCLFLossHead(LossHead):
    # combining binary cross-entropy loss and cross-entropy loss 
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.normalized = False
        self.loss_ce = build_loss_head(cfg.ce, **kwargs)
        self.loss_bce = build_loss_head(cfg.bce, **kwargs)
        self._total_loss = {"ce": 0., "bce": 0.} # record loss
        self.lambd_ce = cfg.lambd_ce
        self.reduce = True 
        # audio -> vision
        layers = list()
        embed_dim = cfg.bce.embed_dim or cfg.bce.width
        sizes = [embed_dim] + list(cfg.layers)
        for i in range(len(sizes) - 2):
            layers.extend([
                LayerNorm(sizes[i]),
                nn.Linear(sizes[i], sizes[i + 1]),
            ])
        layers.extend([
            LayerNorm(sizes[-2]),
            nn.Linear(sizes[-2], sizes[-1], bias=cfg.bias)
        ])
        self.a2v = nn.Sequential(*layers)

    def stats(self, nstep=1, **kwargs):
        msg = " ".join([
            f"{k} {v / nstep:.3f}" for k, v in self._total_loss.items()
        ])
        return msg

    def report(self, gold_file=None):
        report_ce = ""
        if hasattr(self.loss_ce, "x1s") and hasattr(self.loss_ce, "x2s"):
            report_ce = self.loss_ce.report(gold_file=gold_file)
        report_bce = self.loss_bce.report(gold_file=gold_file) 
        return f"{report_ce}\n{report_bce}" 
    
    def forward(self, x1, x2, *args, x3=None, **kwargs):
        """ x1 is (audio) features, x2 is (audio) labels, and x3 is mirror (image) features
        """
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                if x3 is not None:
                    self.loss_ce.infer(self.a2v(x1), x3, *args, **kwargs)
                return self.loss_bce.infer(x1, x2, *args, **kwargs)
            return None
        loss_ce = self.loss_ce(self.a2v(x1), x3, *args, **kwargs)
        loss_bce = self.loss_bce(x1, x2, *args, **kwargs)
        loss = self.lambd_ce * loss_ce + loss_bce
        self._total_loss["ce"] += loss_ce.detach()
        self._total_loss["bce"] += loss_bce.detach()
        return loss

class ImagineAndClassifyLossHead(LossHead):
    # audio-vision contrastive learning task (maybe w/ supervised classification)
    # this is more flexible than `ImaginedCLFLossHead`
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.loss_ce = self.loss_bce = None
        self.normalized = False
        self._total_loss = {}
        if cfg.ce.alive:
            self.loss_ce = build_loss_head(cfg.ce, **kwargs)
            self._total_loss.update({"ce": 0.})
        if cfg.bce.alive:
            self.loss_bce = build_loss_head(cfg.bce, **kwargs)
            self._total_loss.update({"bce": 0.})
        self.lambd_ce = cfg.lambd_ce
        self.reduce = True
        # audio -> vision
        self.a2v = nn.Identity()
        if len(cfg.layers) > 0:
            layers = list()
            embed_dim = cfg.bce.embed_dim or cfg.bce.width
            sizes = [embed_dim] + list(cfg.layers)
            for i in range(len(sizes) - 2):
                layers.extend([
                    LayerNorm(sizes[i]),
                    nn.Linear(sizes[i], sizes[i + 1]),
                ])
            layers.extend([
                LayerNorm(sizes[-2]),
                nn.Linear(sizes[-2], sizes[-1], bias=cfg.bias)
            ])
            self.a2v = nn.Sequential(*layers)

    def stats(self, nstep=1, **kwargs):
        msg = " ".join([
            f"{k} {v / nstep:.3f}" for k, v in self._total_loss.items()
        ])
        return msg

    def report(self, gold_file=None):
        report_ce = report_bce = ""
        if self.loss_ce is not None and hasattr(self.loss_ce, "x1s") and hasattr(self.loss_ce, "x2s"):
            report_ce = self.loss_ce.report(gold_file=gold_file)
        if self.loss_bce is not None:
            report_bce = self.loss_bce.report(gold_file=gold_file)
        return f"{report_ce}\n{report_bce}"

    def infer(self, x1, x2, x3, *args, **kwargs):
        loss_ce = loss_bce = 0.
        if self.loss_ce is not None and x3 is not None:
            loss_ce = self.loss_ce.infer(self.a2v(x1), x3, *args, **kwargs)
        if self.loss_bce is not None:
            loss_bce =  self.loss_bce.infer(x1, x2, *args, **kwargs)
        loss_ce = loss_ce or 0.
        loss_bce = loss_bce or 0.
        return loss_ce + loss_bce

    def forward(self, x1, x2, *args, x3=None, **kwargs):
        """ x1 is (audio) features, x2 is (audio) labels, and x3 is mirror (image) features
        """
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                return self.infer(x1, x2, x3, *args, **kwargs)
            return None

        loss_ce = loss_bce = 0.
        if self.loss_ce is not None and x3 is not None:
            loss_ce = self.loss_ce(self.a2v(x1), x3, *args, **kwargs)
            self._total_loss["ce"] += loss_ce.detach()
        if self.loss_bce is not None:
            loss_bce = self.loss_bce(x1, x2, *args, **kwargs)
            self._total_loss["bce"] += loss_bce.detach()

        loss = self.lambd_ce * loss_ce + loss_bce
        return loss

class LMLossHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.logit_scale = (
            nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if cfg.scaling else
            torch.ones([], requires_grad=False) * np.log(1 / 1)
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.max_len_dec = cfg.max_len_dec
        self.sot_token = "<|startoftext|>"
        self.eot_token = "<|endoftext|>"
        self.reduce = False

    def copy_state_dict(self, state_dict):
        key = "logit_scale"
        new_dict = self.state_dict()
        new_dict.update({key: state_dict[key]})
        self.load_state_dict(new_dict)

    def infer(self, x1, x2, x3, *args, **kwargs):
        if not hasattr(self, "x1s") or not hasattr(self, "x2s") or not hasattr(self, "ids"):
            self.x1s, self.x2s, self.ids = [], [], []
        names = kwargs.get("names", None)
        if names is not None:
            self.ids.extend(names)
        predictions = x3.cpu().tolist()
        for isample, x in enumerate(predictions):
            x = _tokenizer.decode(x)
            x = x.replace(self.sot_token, "")
            eot_pos = x.find(self.eot_token)
            if eot_pos > 0:
                x = x[:eot_pos]
            x = x.strip().split()[:self.max_len_dec]
            x = " ".join(x)
            self.x1s.append({
                "file_name": names[isample],
                "caption_predicted": x,
            })
        return None

    @staticmethod
    def is_json(gold_file):
        with open(gold_file, "r") as fr:
            line = next(fr)
            try:
                line = json.loads(line)
                is_json = isinstance(line, dict)
            except Exception as e:
                is_json = False
        return is_json

    def report(self, gold_file=None):
        assert gold_file is not None, f"please provide the right gold file: `{gold_file}`."
        references = list()
        nsample = len(self.x1s)

        # csv (Clotho) or json (AudioCaps)
        if not self.is_json(gold_file):
            with open(gold_file, "r") as fr:
                fr = csv.DictReader(fr)
                for iline, line in enumerate(fr):
                    references.append(line)
                    if iline + 1 >= nsample:
                        break
        else:
            with open(gold_file, "r") as fr:
                for iline, line in enumerate(fr):
                    record = json.loads(line)
                    item = {
                        f"caption_{i + 1}": caption for i, caption in enumerate(record["captions"])
                    }
                    item["file_name"] = record["id"]
                    references.append(item)
                    if iline + 1 >= nsample:
                        break

        key = "file_name"
        ref_keys = [r[key] for r in references]
        ret_keys = [r[key] for r in self.x1s]
        f_t = set(ref_keys) - set(ret_keys)
        t_f = set(ret_keys) - set(ref_keys)
        #print(f"{f_t}\n{t_f}")

        try:
            msg = []
            results = ac_metric(self.x1s, references)
            for k, v in results.items():
                msg.append(f"{k}: {v['score']:.3f}")
            msg = " ".join(msg)
        except Exception as e:
            msg = f"failed to evaluate the model: {e}"

        del self.x1s, self.x2s, self.ids
        report = f"{msg}"
        return report

    def forward(self, x1, x2, x3, *args, **kwargs):
        # x1: logits; x2: word seqs; predictions
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                return self.infer(x1, x2, x3, *args, **kwargs)
            return None
        # cosine similarity as logits
        x1 = x1.reshape(-1, x1.shape[-1])
        logit_scale = self.logit_scale.exp()
        logits_per_x1 = logit_scale * x1
        # cross entropy loss
        labels = x2.reshape(-1)
        loss_mean_x1 = self.loss_fn(logits_per_x1, labels)
        loss = loss_mean_x1
        return loss
