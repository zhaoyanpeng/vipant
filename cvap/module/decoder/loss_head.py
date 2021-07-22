from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from collections import defaultdict
from clip import LayerNorm, Transformer, ModifiedResNet, VisualTransformer  

LOSS_HEADS_REGISTRY = Registry("LOSS_HEADS")
LOSS_HEADS_REGISTRY.__doc__ = """
Registry for image encoders.
"""

def build_loss_head(cfg, **kwargs):
    return LOSS_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

class LossHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = False

    def copy_state_dict(self, state_dict): 
        pass

    def infer(self, x1, x2, *args, **kwargs):
        if not hasattr(self, "x1s") or not hasattr(self, "x2s") or not hasattr(self, "ids"): 
            self.x1s, self.x2s, self.ids = [], [], []
        # normalized features
        if not kwargs.get("normalized", False):
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            x2 = x2 / x2.norm(dim=-1, keepdim=True)
        self.x1s.append(x1)
        self.x2s.append(x2)
        names = kwargs.get("names", None)
        if names is not None:
            self.ids.extend(names)
        return None  

    def _gold_cluster(self, gold_file, nsample, verbose=False):
        sample_by_classname = defaultdict(list)
        classname_by_sample = defaultdict(str)
        with open(gold_file, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                if iline + 1 > nsample:
                    break
                key = " ".join(record["labels"])
                sample_by_classname[key].append(record["id"])
                classname_by_sample[record["id"]] = key
        if verbose:
            items = list(sample_by_classname.items())
            items.sort(key=lambda x: -len(x[1]))
            for k, v in items:
                print(k, len(v))
            print(f"total {len(sample_by_classname)} groups")
        return sample_by_classname, classname_by_sample
        
    def report(self, gold_file=None):
        x1s = torch.cat(self.x1s)
        x2s = torch.cat(self.x2s)
        nsample = x1s.shape[0]
        labels = torch.arange(nsample, device=x1s.device).unsqueeze(-1)
        # x1 -> x2
        x12 = x1s @ x2s.t()
        ind_12 = x12.argsort(descending=True)
        r12 = torch.where(ind_12 == labels)[1]
        
        t12_1 = torch.where(r12 < 1)[0].shape[0] / nsample * 100. 
        t12_5 = torch.where(r12 < 5)[0].shape[0] / nsample * 100. 

        # x2 -> x1
        x21 = x2s @ x1s.t()
        ind_21 = x21.argsort(descending=True)
        r21 = torch.where(ind_21 == labels)[1]

        t21_1 = torch.where(r21 < 1)[0].shape[0] / nsample * 100. 
        t21_5 = torch.where(r21 < 5)[0].shape[0] / nsample * 100. 

        # stats per class
        if gold_file is not None:
            nsample = x1s.size(0) 
            sample_by_classname, classname_by_sample = self._gold_cluster(gold_file, nsample)

            def topk_overlap(x, k=1):
                """ x has to be a 2d matrix: (sample_idx, sorted_sample_idx)
                """
                indice = x[:, :k]
                stats = defaultdict(dict) # {cls_name: {sample: num_correct}} 
                for idx, neighbors in enumerate(indice):
                    sample = self.ids[idx]
                    classname = classname_by_sample[sample]
                    true_neighbors = sample_by_classname[classname]
                    sample_stat = stats.get(classname, {})
                    this_stat = sample_stat.get(sample, [0] * 2)
                    for neighbor in neighbors:
                        neighbor_sample = self.ids[neighbor]
                        if neighbor_sample in true_neighbors:
                            this_stat[0] += 1 
                    sample_stat[sample] = this_stat 
                    stats[classname] = sample_stat
                return stats

            def pnr(stats, k=1, msg=""):
                """ P: relevant & retrieved / retrieved
                    R: relevant & retrieved / relevant 
                """
                p, r = 0., 0. # overall
                p_cls, r_cls = 0., 0. # in-class
                pnr_by_class = defaultdict(list)
                nclass = len(sample_by_classname)
                for classname, class_stats in stats.items():
                    pnr_cls = pnr_by_class.get(classname, [0] * 3)
                    nrelevant = len(sample_by_classname[classname]) 
                    for sample, sample_stats in class_stats.items():
                        tp = sample_stats[0]
                        this_p = tp / k 
                        this_r = tp / nrelevant
                        p += this_p 
                        r += this_r
                        pnr_cls[0] += this_p  
                        pnr_cls[1] += this_r 
                    pnr_cls[0] /= nrelevant 
                    pnr_cls[1] /= nrelevant 
                    pnr_by_class[classname] = pnr_cls
                    p_cls += pnr_cls[0]
                    r_cls += pnr_cls[1]
                p = (p / nsample) * 100
                r = (r / nsample) * 100 
                p_cls = (p_cls / nclass) * 100 
                r_cls = (r_cls / nclass) * 100 
                return f"{msg}: P@{k} {p:2.2f} R@{k} {r:2.2f} mAP {p_cls:2.2f} mAR {r_cls:2.2f}"

            # x1 -> x2
            stats_12 = topk_overlap(ind_12, k=1) 
            msg_12 = pnr(stats_12, k=1, msg="I->A")

            # x2 -> x1
            stats_21 = topk_overlap(ind_21, k=1) 
            msg_21 = pnr(stats_21, k=1, msg="A->I")

        del self.x1s, self.x2s, self.ids
        report = (
            f"\n{msg_12} {msg_21}\n" +
            f"I->A: t1 = {t12_1:2.2f} t5 = {t12_5:2.2f} " + 
            f"A->I: t1 = {t21_1:2.2f} t5 = {t21_5:2.2f} @ {nsample}" 
        )
        return report

@LOSS_HEADS_REGISTRY.register()
class CELossHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = False 
    
    def copy_state_dict(self, state_dict): 
        key = "logit_scale"
        new_dict = self.state_dict()
        new_dict.update({key: state_dict[key]})
        self.load_state_dict(new_dict)

    def forward(self, x1, x2, *args, **kwargs):
        if not self.training:
            if not dist.is_initialized() or dist.get_rank() == 0:
                return self.infer(x1, x2, *args, **kwargs)
            return None 
        # normalized features
        if not kwargs.get("normalized", False):
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            x2 = x2 / x2.norm(dim=-1, keepdim=True)
        #print(f"{threading.current_thread().ident} loss --{kwargs.get('normalized', False)}")
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_x1 = logit_scale * x1 @ x2.t()
        logits_per_x2 = logit_scale * x2 @ x1.t()
        # cross entropy loss 
        labels = torch.arange(x1.shape[0], device=x1.device)
        loss_mean_x1 = self.loss_fn(logits_per_x1, labels)
        loss_mean_x2 = self.loss_fn(logits_per_x2, labels)
        loss = loss_mean_x1 + loss_mean_x2
        return loss

@LOSS_HEADS_REGISTRY.register()
class BarlowLossHead(LossHead):
    # see Barlow Twins: https://arxiv.org/abs/2103.03230 
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(cfg.embed_dim, affine=False) 
        self.off_weight = cfg.off_weight
        self.reduce = True 
    
    @staticmethod
    def loss_fn(c):
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.masked_select( # more memory-efficient?
            ~torch.eye(c.size(-1), device=c.device, dtype=torch.bool)
        ).pow_(2).sum()
        return on_diag, off_diag
    
    def forward(self, x1, x2, *args, **kwargs):
        if not self.training:
            return self.infer(x1, x2)
        x1, x2 = self.bn(x1), self.bn(x2)
        c = x1.t() @ x2
        c.div_(x1.size(0))
        on_diag, off_diag = self.loss_fn(c)
        loss = on_diag + self.off_weight * off_diag
        return loss

@LOSS_HEADS_REGISTRY.register()
class ClassificationHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        assert "output_dim" in kwargs, f"`the label number` is not found in `kwargs`"
        nlabel = kwargs["output_dim"]
        self.linear = nn.Sequential(
            LayerNorm(cfg.embed_dim), 
            nn.Linear(cfg.embed_dim, nlabel)
        )  
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = False 
    
    def copy_state_dict(self, state_dict): 
        key = "logit_scale"
        new_dict = self.state_dict()
        new_dict.update({key: state_dict[key]})
        self.load_state_dict(new_dict)

    def infer(self, x1, x2, *args, **kwargs):
        if not hasattr(self, "x1s") or not hasattr(self, "x2s") or not hasattr(self, "ids"): 
            self.x1s, self.x2s, self.ids = [], [], []
        logits_per_x1 = self.linear(x1)
        predictions = logits_per_x1.argmax(-1) 
        self.x1s.append(predictions)
        self.x2s.append(x2)
        names = kwargs.get("names", None)
        if names is not None:
            self.ids.extend(names)
        return None  

    def report(self, gold_file=None):
        x1s = torch.cat(self.x1s)
        x2s = torch.cat(self.x2s)
        nsample = len(x1s)
        precision = (x1s == x2s).sum() / nsample * 100.

        del self.x1s, self.x2s, self.ids
        report = (
            f"A->T: p1 = {precision:2.2f} @ {nsample}" 
        )
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
        loss_mean_x1 = self.loss_fn(logits_per_x1, x2)
        return loss_mean_x1
