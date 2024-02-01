import argparse

import numpy as np
import yaml

from config.get_cfg import get_cfg
from detector import get_ood_detector
from model import get_model
from utils.dataset import get_dataloaders
from utils.metrics import get_measures_from_pred


def main(args):
    cfg = get_cfg(args.experiment, args.model)

    # Load config of datasets
    with open("./config/datasets.yaml", 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    # Load model
    model = get_model(cfg, resume=cfg.resume)

    # Get OOD Detaction
    if cfg.exp_space == "CIFAR10":
        benchmark_id = 0
    elif cfg.exp_space == "CIFAR100":
        benchmark_id = 1
    else:
        benchmark_id = 2
    ood_detector = get_ood_detector(args.detector, benchmark_id,
                                    use_surrogate=args.use_surrogate,
                                    is_gaussian=args.is_gaussian, use_real=args.use_real,
                                    use_ood_score=args.use_ood_score)
    ood_detector.setup(model, cfg, dataset_cfg)

    # Get dataset loaders
    id_loader, near_ood_loaders, far_ood_loaders = get_dataloaders(
        ["id", "ood"],
        cfg, dataset_cfg, batch_size=args.batch_size,
        num_workers=16)

    # Get ID predictions
    id_pred_list, id_conf_list, id_label_list = ood_detector.inference(model, id_loader)

    # OOD datasets
    ood_dict = {
        "near": near_ood_loaders,
        "far": far_ood_loaders
    }
    all_results_summary = []
    ood_task_summary = []

    for near_or_far, loaders in ood_dict.items():
        for name, loader in loaders.items():
            print(f"----------{near_or_far}, {name}-----------")
            ood_pred_list, ood_conf_list, ood_label_list = ood_detector.inference(model, loader)

            # Get metrics
            res = get_measures_from_pred(ood_conf_list, id_conf_list)
            all_results_summary.append(res[3])
            ood_task_summary.append(res)

    # Get mean performance
    auroc = np.mean([x[0] for x in ood_task_summary])
    aupr = np.mean([x[1] for x in ood_task_summary])
    fpr = np.mean([x[2] for x in ood_task_summary])
    print(f"----------Mean-----------")
    print('FPR95:\t\t\t{:.2f}'.format(100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    res = "{:.2f} & {:.2f}".format(100 * fpr, 100 * auroc)
    all_results_summary.append(res)

    print("Result summary:", " & ".join(all_results_summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Evaluate different OOD detection methods.')
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch Size. It should have no influence on results but on the IO speed.")
    parser.add_argument("-d", "--detector", default="msp", help="The OOD detection method.")
    parser.add_argument("-e", "--experiment", default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'],
                        help="The benchmark name.")
    parser.add_argument("-m", "--model", default="rn50",
                        choices=['rn50', 'mb', 'vit_b', 'vit_l', 'swin_s', 'swin_b',
                                 'mix_b', 'mix_l', 'dense', 'mix_n'],
                        help='The model for evaluation.')
    parser.add_argument("-s", "--use_surrogate", action='store_true',
                        help="Use surrogate to estimate parameters in methods. See Appendix.B.")
    parser.add_argument("-p", "--p", type=int, default=50,
                        help="Hyperparameters in prior methods. See Appendix.A.")
    parser.add_argument("-g", "--is_gaussian", action='store_true',
                        help="Use Gaussian surrogate to estimate parameters in methods. See Appendix.B.")
    parser.add_argument("-r", "--use_real", action='store_true',
                        help="Use read OOD data to estimate parameters in methods. See Appendix.B.")
    parser.add_argument("-o", "--use_ood_score", default="vanilla",
                        choices=['vanilla', 'energy'],
                        help="The vanilla method denoted “Ours (V)”, based on the maximum logit score, and there is a variant based on energy scores.")

    main(parser.parse_args())
