import os
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
from datetime import datetime
from dataclasses import replace
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.logging.core import AttackResult
from distribution_inference.utils import flash_utils

def plot_gaussians(m1, m2, s1, s2, threshold): 
    x_min, x_max = threshold-.1, threshold+.1
    x = np.linspace(x_min, x_max, 400)
    pdf1 = norm.pdf(x, loc=m1, scale=s1)
    pdf2 = norm.pdf(x, loc=m2, scale=s2)

    plt.figure(figsize=(6, 4))
    plt.plot(x, pdf1, linestyle='-', linewidth=2, label='Gaussian 1')
    plt.plot(x, pdf2, linestyle='--', linewidth=2, label='Gaussian 2')
    plt.fill_between(x, pdf1, alpha=0.2)
    plt.fill_between(x, pdf2, alpha=0.2)

    plt.xlim(x_min, x_max)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Zoomed-in around Gaussian Intersection')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/ujx4ab/ondemand/dissecting_dist_inf/gaussian_errors_poisoining.png")

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--en", type=str, required=True)
    parser.add_argument("--load_config", type=Path, required=True)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    attack_config: AttackConfig = AttackConfig.load(args.load_config, drop_extra_fields=False)
    bb_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config

    flash_utils(attack_config)
    logger = AttackResult(args.en, attack_config)

    train_adv_config_clean = get_train_config_for_adv(train_config, attack_config)
    ds_wrapper_class = get_dataset_wrapper(data_config.name)
    
    alphas = attack_config.values

    # get victim models
    data_config_adv, data_config_vic = get_dfs_for_victim_and_adv(data_config)
    poison_rate = bb_config.poison_config.poison_rates[0]
    train_adv_config_poison = replace(train_adv_config_clean, data_config=replace(train_adv_config_clean.data_config, poison_rate=poison_rate))
    data_config_vic_poison = replace(data_config_vic, poison_rate=poison_rate)
    ds_vic_poison = ds_wrapper_class(data_config_vic_poison)

    vic_models_poisoned = ds_vic_poison.get_models(
        train_adv_config_poison,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=attack_config.train_config.save_every_epoch,
        model_arch=attack_config.victim_model_arch
    )

    # loop through alphas and get adv baseline and victim performance
    adv_clean_errors = {}
    adv_poisoned_errors = {}
    vic_poisoned_errors = {}

    for alpha in alphas:
        # clean config
        data_config_adv = replace(data_config_adv, value=alpha, poison_rate=0.0)
        ds_adv_clean = ds_wrapper_class(data_config_adv)

        # poison config
        data_config_poison = replace(data_config_adv, poison_rate=poison_rate)
        ds_adv_poison = ds_wrapper_class(data_config_poison)

        adv_models_clean = ds_adv_clean.get_models(
            train_adv_config_clean,
            n_models=bb_config.num_adv_models,
            model_arch=attack_config.adv_model_arch,
            on_cpu=attack_config.on_cpu,
            target_epoch=attack_config.adv_target_epoch
        )

        adv_models_poisoned = ds_adv_poison.get_models(
            train_adv_config_poison,
            n_models=bb_config.num_adv_models,
            model_arch=attack_config.adv_model_arch,
            on_cpu=attack_config.on_cpu,
            target_epoch=attack_config.adv_target_epoch
        )

        print(f"Run inference with models_clean & models_pois on alpha={alpha}, poisoning={ds_adv_clean.data_config.poison_rate}")
        adv_preds, vic_preds, ground_truth, _, property_labels = get_vic_adv_preds_on_distr(
            models_vic=(vic_models_poisoned, vic_models_poisoned),
            models_adv=(adv_models_clean, adv_models_poisoned),
            ds_obj=ds_adv_clean,
            batch_size=bb_config.batch_size,
            preload=bb_config.preload,
            multi_class=bb_config.multi_class,
            make_processed_version=attack_config.adv_processed_variant,
            return_props_labels=True
        )

        prop_idx = property_labels.squeeze() == 1
        gt = ground_truth[prop_idx].squeeze()

        adv_preds_clean = adv_preds.preds_property_1[:, prop_idx]
        adv_preds_poisoned = adv_preds.preds_property_2[:, prop_idx]
        vic_preds_poisoned = vic_preds.preds_property_1[:, prop_idx]
        #preds prop 2 is the same as 1 for vic

        adv_clean_errors[alpha] = ((adv_preds_clean - gt)**2).mean(axis=1)
        adv_poisoned_errors[alpha] = ((adv_preds_poisoned - gt)**2).mean(axis=1)
        vic_poisoned_errors[alpha] = ((vic_preds_poisoned - gt)**2).mean(axis=1)

        print(f"alpha: {alpha}")
        print(adv_clean_errors[alpha].mean())
        print(adv_poisoned_errors[alpha].mean())
        print(vic_poisoned_errors[alpha].mean())

        logger.add_results(attack_type, prop_value,
            adv_clean_errors[alpha].mean(), vic_poisoned_errors[alpha].mean())
        logger.save()

    def gaussian_threshold_scipy(x1, x2):
        m1, s1 = norm.fit(x1)
        m2, s2 = norm.fit(x2)
        
        if np.isclose(s1, s2):
            print("WARNING: difference in distrbutions is very small ... using midpoint")
            return 0.5 * (m1 + m2)

        A = s2**2 - s1**2
        B = (m1 - m2)**2 + 2 * A * np.log(s2 / s1)
        sqrt_term = np.sqrt(B) * s1 * s2
        t_vals = [
            (m1 * s2**2 - m2 * s1**2 + sqrt_term) / A,
            (m1 * s2**2 - m2 * s1**2 - sqrt_term) / A
        ]
        low, high = sorted((m1, m2))
        for t in t_vals:
            if low <= t <= high:
                return t
        return 0.5 * (m1 + m2)

    thresholds = {}
    for α in alphas:
        clean_err = adv_clean_errors[α]
        poisoned_err = adv_poisoned_errors[α]
        thresholds[α] = gaussian_threshold_scipy(clean_err, poisoned_err)

    print("Optimal thresholds per alpha:", thresholds)
        
    predicted_alpha = {}
    for true_α in alphas:
        vic_errs = vic_poisoned_errors[true_α]
        vic_mean = np.mean(vic_errs)
        # pick the α whose threshold is closest to this mean
        closest_α = min(thresholds.keys(), key=lambda a: abs(vic_mean - thresholds[a]))
        predicted_alpha[true_α] = closest_α

    print("Thresholds per α:", thresholds)
    print("Predicted α_for_each_true_α:", predicted_alpha)

