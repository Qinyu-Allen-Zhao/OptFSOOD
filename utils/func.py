import os

import joblib
import numpy as np
import torch
import tqdm
import yaml

from config.get_cfg import get_cfg
from model import get_model
from utils.dataset import get_dataloaders, get_surrogate_data_loader, get_real_ood_val_loader
from utils.metrics import cal_auroc_from_conf


@torch.no_grad()
def inference_with_feat(model, dataloader, num_iter=None):
    pred_list, conf_list, label_list, feat_list = [], [], [], []
    if num_iter is None:
        num_iter = len(dataloader)
    else:
        num_iter = min(len(dataloader), num_iter)

    iterator = iter(dataloader)
    for i in tqdm.trange(num_iter):
        data, label = next(iterator)
        data = data.cuda()
        output, feat = model(data, return_feature=True)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)

        pred_list.append(pred.detach().cpu().numpy())
        conf_list.append(conf.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        feat_list.append(feat.detach().cpu().numpy())

        # if i > 20:
        #     break

    pred_list = np.concatenate(pred_list, axis=0).astype(int)
    conf_list = np.concatenate(conf_list, axis=0)
    label_list = np.concatenate(label_list, axis=0).astype(int)
    feat_list = np.concatenate(feat_list, axis=0)

    return pred_list, conf_list, label_list, feat_list


@torch.no_grad()
def clip_feature_only_pred_class(model, id_feat, ood_feat, id_pred, ood_pred):
    from utils.metrics import get_measures_from_pred

    W, b = model.get_fc()
    W = W.detach().cpu()
    b = b.detach().cpu()

    id_feat_t = torch.from_numpy(id_feat)
    ood_feat_t = torch.from_numpy(ood_feat)

    id_output = []
    for i in tqdm.trange(len(id_feat_t)):
        id_weight_x_feat = W * id_feat_t[[i], :]
        id_weight_x_feat = id_weight_x_feat.clip(max=0.02)
        res = id_weight_x_feat.sum(dim=1) + b
        id_output.append(res[None, :])
    id_conf = torch.logsumexp(torch.cat(id_output, dim=0), dim=1)

    ood_output = []
    for i in tqdm.trange(len(ood_feat_t)):
        ood_weight_x_feat = W * ood_feat_t[[i], :]
        ood_weight_x_feat = ood_weight_x_feat.clip(max=0.02)
        res = ood_weight_x_feat.sum(dim=1) + b
        ood_output.append(res[None, :])
    ood_conf = torch.logsumexp(torch.cat(ood_output, dim=0), dim=1)

    get_measures_from_pred(ood_conf, id_conf)


@torch.no_grad()
def get_predict_with_clipped_feat(model, id_feat, ood_feats, process_feat, args, get_all_aurocs=False):
    auroc_list = []
    for ood_feat in ood_feats:
        id_feat_p = process_feat(id_feat, args).cuda()
        id_output = model.fc(id_feat_p)
        id_conf = torch.logsumexp(id_output, dim=1).cpu().numpy()

        ood_feat_p = process_feat(ood_feat, args).cuda()
        ood_output = model.fc(ood_feat_p)
        ood_conf = torch.logsumexp(ood_output, dim=1).cpu().numpy()

        auroc_list.append(cal_auroc_from_conf(id_conf, ood_conf) * 100)

    if get_all_aurocs:
        return auroc_list
    else:
        return -np.mean(auroc_list)


@torch.no_grad()
def get_all_features(id_dataset='cifar100', model_type='dense'):
    cfg = get_cfg(id_dataset, model_type)

    # Load model
    model = get_model(cfg, resume=cfg.resume)
    stored = f"./output/features/{id_dataset}_{model_type}_features_all_datasets.pkl"

    if os.path.isfile(stored):
        # Load stored files
        res = joblib.load(stored)
    else:
        # Load config of datasets
        with open("./config/datasets.yaml", 'r') as f:
            dataset_cfg = yaml.safe_load(f)

        # Get dataset loaders
        id_loader, near_ood_loaders, far_ood_loaders = get_dataloaders(
            ["id", "ood"],
            cfg, dataset_cfg, batch_size=128)

        # ID predictions
        id_pred, id_conf, id_label, id_feat = inference_with_feat(model, id_loader, num_iter=None)
        res = {"ID": (id_pred, id_conf, id_label, id_feat)}
        # OOD predictions
        for dname, ood_loader in near_ood_loaders.items():
            res[dname] = inference_with_feat(model, ood_loader, num_iter=None)
        for dname, ood_loader in far_ood_loaders.items():
            res[dname] = inference_with_feat(model, ood_loader, num_iter=None)

        joblib.dump(res, stored)

    return model, res


@torch.no_grad()
def get_surrogate_ood_features(cfg, model, dataset_cfg, is_gaussian=True, use_real=False):
    if use_real:
        # near_ood_loaders, far_ood_loaders = get_dataloaders(
        #     ["ood"], cfg, dataset_cfg, batch_size=128)
        # pred, feat = [], []
        # for dname, ood_loader in near_ood_loaders.items():
        #     p, _, _, f = inference_with_feat(model, ood_loader, num_iter=None)
        #     pred.append(p)
        #     feat.append(f)
        # for dname, ood_loader in far_ood_loaders.items():
        #     p, _, _, f = inference_with_feat(model, ood_loader, num_iter=None)
        #     pred.append(p)
        #     feat.append(f)
        # pred = np.concatenate(pred, axis=0)
        # feat = np.concatenate(feat, axis=0)
        data_loader = get_real_ood_val_loader(cfg, dataset_cfg)
    else:
        # Get random dataset loaders
        data_loader = get_surrogate_data_loader(cfg, is_gaussian)

    # predictions
    pred, _, _, feat = inference_with_feat(model, data_loader, num_iter=None)

    return pred, feat


@torch.no_grad()
def get_feature_train(model, cfg, dataset_cfg):
    stored = f"./output/features/{cfg.dataset.id_dataset}_{cfg.model_type}_train_features.npz"
    if os.path.isfile(stored):
        train_info = np.load(stored)
        feature_id_train = train_info['feature']
        pred_id_train = train_info['pred']
    else:
        train_loader = get_dataloaders(["train"], cfg, dataset_cfg, batch_size=512, num_workers=16)[0]

        print('Extracting id training feature')
        feature_id_train, pred_id_train = [], []
        for data, label in tqdm.tqdm(train_loader, desc='Eval: ', position=0, leave=True):
            data = data.cuda().float()
            logit, feature = model(data, return_feature=True)
            score = torch.softmax(logit, dim=1)
            _, pred = torch.max(score, dim=1)

            feature_id_train.append(feature.cpu().numpy())
            pred_id_train.append(pred.cpu().numpy())

        feature_id_train = np.concatenate(feature_id_train, axis=0)
        pred_id_train = np.concatenate(pred_id_train, axis=0)
        np.savez(stored, feature=feature_id_train, pred=pred_id_train)

    print(feature_id_train.shape)
    print(pred_id_train.shape)

    return feature_id_train, pred_id_train


def get_fxw(w, pred, feature, left_boundary, width, show_progress=False):
    lc = w[pred] * feature
    fxws = []
    x_range = tqdm.tqdm(left_boundary) if show_progress else left_boundary
    for x in x_range:
        mask = (feature >= x) & (feature < x + width)
        feat_masked = mask * lc
        fxw = torch.sum(feat_masked, dim=1, keepdim=True)
        fxws.append(fxw)
    fxws = torch.cat(fxws, dim=1)

    return fxws
