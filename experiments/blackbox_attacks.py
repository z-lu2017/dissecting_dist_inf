from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_output_distributions(
    property_name,
    predictions_groups,  # list of prediction‐lists: each [M1@r0, M1@r1, M2@r0, M2@r1]
    ratios,              # tuple of ratio keys, e.g. ('a0','a1')
    model_names,         # ['M1','M2']
    folder_path,
    bins=50,
    kde_grid_points=300,
    colors=None
):
    os.makedirs(folder_path, exist_ok=True)

    # Flatten all preds to find global x and y limits
    all_preds = np.concatenate([np.concatenate(g).ravel() for g in predictions_groups])
    x_min, x_max = all_preds.min(), all_preds.max()
    grid = np.linspace(x_min, x_max, kde_grid_points)

    # Compute global y_max
    y_max = 0
    for group in predictions_groups:
        for preds in group:
            dens_hist, _ = np.histogram(preds, bins=bins, density=True)
            y_max = max(y_max, dens_hist.max())
            y_max = max(y_max, gaussian_kde(preds.ravel())(grid).max())
    y_max *= 1.05
    y_min = 0

    # Default colors if not provided
    if colors is None:
        colors = {model_names[0]: 'C0', model_names[1]: 'C1'}

    # Define the four index‐pairs to compare:
    # 1&3 → indices (0,2), 2&3 → (1,2), 1&4 → (0,3), 2&4 → (1,3)
    compare_idxs = [(0,2), (1,2), (0,3), (1,3)]

    for group_idx, group in enumerate(predictions_groups):
        test_ratio = ratios[group_idx]

        for idx_a, idx_b in compare_idxs:
            preds_a = group[idx_a].ravel()
            preds_b = group[idx_b].ravel()

            def make_label(idx):
                model_i = idx // len(ratios)
                ratio_i = idx % len(ratios)
                return f"{model_names[model_i]} (trained on {ratios[ratio_i]})"

            label_a = make_label(idx_a)
            label_b = make_label(idx_b)

            plt.figure(figsize=(8,5))
            # histogram + KDE for A
            plt.hist(preds_a, bins=bins, density=True, histtype='step',
                     linewidth=1.5, color=colors[label_a.split()[0]], label=f"{label_a} hist")
            plt.plot(grid, gaussian_kde(preds_a)(grid), linewidth=2,
                     color=colors[label_a.split()[0]], label=f"{label_a} KDE")

            # histogram + KDE for B
            plt.hist(preds_b, bins=bins, density=True, histtype='step',
                     linewidth=1.5, color=colors[label_b.split()[0]], label=f"{label_b} hist")
            plt.plot(grid, gaussian_kde(preds_b)(grid), linewidth=2,
                     color=colors[label_b.split()[0]], label=f"{label_b} KDE")

            plt.xlabel('Model output')
            plt.ylabel('Density')
            plt.title(f"{property_name}: {label_a.split()[0]} vs {label_b.split()[0]} on {test_ratio}")
            plt.legend()
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.tight_layout()

            fname = f"{property_name}_{label_a.replace(' ','')}_vs_{label_b.replace(' ','')}_on_{test_ratio}.png"
            save_path = os.path.join(folder_path, fname)
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    parser.add_argument(
        "--prop", help="Property for which to run the attack",
        type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Loading config from:", args.load_config)

    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Use given prop (if given) or the one in the config
    if args.prop is not None:
        attack_config.train_config.data_config.prop = args.prop

    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(attack_config)
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)
    #print("what is ds_wrapper_class = ", ds_wrapper_class)
    #time.sleep(100)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # data configs are the same, we just have adv and victim versions (alpha0)
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    
    # alpha0 data config for vic
    ds_vic_1 = ds_wrapper_class(
        data_config_vic_1,
        skip_data=True,
        label_noise=train_config.label_noise,
        epoch=attack_config.train_config.save_every_epoch)
    # alpha0 data config for adv
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)

    def single_evaluation(models_1_path=None, models_2_paths=None):
        # Load vic trained on alpha0
        # initial model if no trained before
        models_vic_1 = ds_vic_1.info_object.get_model()
        '''
        models_vic_1 = ds_vic_1.get_models( # !!! value = 1.0, split = victim
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            epochwise_version=attack_config.train_config.save_every_epoch,
            model_arch=attack_config.victim_model_arch,
            custom_models_path=models_1_path)
        '''
        if type(models_vic_1) == tuple:
                models_vic_1 = models_vic_1[0]

        # loop through alpha1 values to test
        for prop_value in attack_config.values:
            data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
                data_config, prop_value=prop_value)

            # datasets for alpha1
            ds_vic_2 = ds_wrapper_class(
                data_config_vic_2, skip_data=True,
                label_noise=train_config.label_noise,
                epoch=attack_config.train_config.save_every_epoch)
            ds_adv_2 = ds_wrapper_class(data_config_adv_2)
            '''
            models_vic_2 = ds_vic_2.get_models( # !!! value = 1.0, split = victim
                train_config, 
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                epochwise_version=attack_config.train_config.save_every_epoch,
                model_arch=attack_config.victim_model_arch,
                custom_models_path=models_2_paths[i] if models_2_paths else None)
            '''
            models_vic_2 = ds_vic_2.info_object.get_model()
            if type(models_vic_2) == tuple:
                models_vic_2 = models_vic_2[0]

            # each try we used different set of adv models
            for t in range(attack_config.tries):
                print("{}: trial {}".format(prop_value, t))
                '''
                models_adv_1 = ds_adv_1.get_models( # !!! value = 1.0, split = adv
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch,
                    target_epoch = attack_config.adv_target_epoch)
                '''
                models_adv_1 = ds_adv_1.info_object.get_model()
                if type(models_adv_1) == tuple:
                    models_adv_1 = models_adv_1[0]
                '''
                models_adv_2 = ds_adv_2.get_models( # !!! value = 1.0, split = adv
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch,
                    target_epoch = attack_config.adv_target_epoch)
                '''
                models_adv_2 = ds_adv_2.info_object.get_model()
                if type(models_adv_2) == tuple:
                    models_adv_2 = models_adv_2[0]


                # preds_adv_on_1: adv (trained on alpha1), adv (train on alpha2) prediction on alpha1
                # preds_vic_on_1: vic (trained on alpha1), vic (train on alpha2) prediction on alpha1
                # ground_truth_1: true labels of ds_adv_1
                preds_adv_on_1, preds_vic_on_1, ground_truth_1, not_using_logits, property_labels_on_1 = get_vic_adv_preds_on_distr(
                    models_vic=(models_vic_1, models_vic_2),
                    models_adv=(models_adv_1, models_adv_2),
                    ds_obj=ds_adv_1,
                    batch_size=bb_attack_config.batch_size,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    preload=bb_attack_config.preload,
                    multi_class=bb_attack_config.multi_class,
                    make_processed_version=attack_config.adv_processed_variant, 
                    return_props_labels=attack_config.black_box.return_props_labels, 
                    regression_task=bb_attack_config.regression_task,
                    use_hidden_state=attack_config.use_hidden_state,
                    use_cell_state=attack_config.use_cell_state
                )

                # preds_adv_on_2: adv (trained on alpha1), adv (train on alpha2) prediction on alpha2
                # preds_vic_on_2: vic (trained on alpha1), vic (train on alpha2) prediction on alpha2
                # ground_truth_2 = true labels of ds_adv_2
                preds_adv_on_2, preds_vic_on_2, ground_truth_2, _, property_labels_on_2 = get_vic_adv_preds_on_distr(
                    models_vic=(models_vic_1, models_vic_2),
                    models_adv=(models_adv_1, models_adv_2),
                    ds_obj=ds_adv_2,
                    batch_size=bb_attack_config.batch_size,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    preload=bb_attack_config.preload,
                    multi_class=bb_attack_config.multi_class,
                    make_processed_version=attack_config.adv_processed_variant,
                    return_props_labels=attack_config.black_box.return_props_labels, 
                    regression_task=bb_attack_config.regression_task,
                    use_hidden_state=attack_config.use_hidden_state,
                    use_cell_state=attack_config.use_cell_state
                )

                # # unpack your prediction objects into raw arrays
                # preds_adv1_1 = preds_adv_on_1.preds_property_1  # trained on a0, tested on a0
                # preds_adv2_1 = preds_adv_on_1.preds_property_2  # trained on a1, tested on a0
                # preds_vic1_1 = preds_vic_on_1.preds_property_1  # trained on a0, tested on a0
                # preds_vic2_1 = preds_vic_on_1.preds_property_2  # trained on a1, tested on a0
                # predictions_list1 = [preds_adv1_1, preds_adv2_1, preds_vic1_1, preds_vic2_1]

                # # repeat for the second probe set
                # preds_adv1_2 = preds_adv_on_2.preds_property_1
                # preds_adv2_2 = preds_adv_on_2.preds_property_2
                # preds_vic1_2 = preds_vic_on_2.preds_property_1
                # preds_vic2_2 = preds_vic_on_2.preds_property_2
                # predictions_list2 = [preds_adv1_2, preds_adv2_2, preds_vic1_2, preds_vic2_2]

                # plot_output_distributions(
                #     property_name=train_config.data_config.prop,
                #     ratios=(str(1-data_config.value), str(1-prop_value)),
                #     predictions_groups=[predictions_list1, predictions_list2],
                #     model_names=["M1_DisjointWT", "M2_DisjointWT"],
                #     folder_path="/home/ujx4ab/ondemand/dissecting_dist_inf/experiments/log/",
                # )

                preds_adv = PredictionsOnDistributions(
                    preds_on_distr_1=preds_adv_on_1,
                    preds_on_distr_2=preds_adv_on_2 
                )
                preds_vic = PredictionsOnDistributions(
                    preds_on_distr_1=preds_vic_on_1,
                    preds_on_distr_2=preds_vic_on_2
                )

                # For each requested attack
                for attack_type in bb_attack_config.attack_type:
                    # Create attacker object
                    attacker_obj = get_attack(attack_type)(bb_attack_config)
                    # Launch attack
                    result = attacker_obj.attack(
                        preds_adv, preds_vic,
                        ground_truth=(ground_truth_1, ground_truth_2),
                        calc_acc=calculate_accuracies,
                        epochwise_version=attack_config.train_config.save_every_epoch,
                        not_using_logits=not_using_logits,
                        regression_task=bb_attack_config.regression_task,
                    )

                    logger.add_results(attack_type, prop_value,
                                       result[0][0], result[1][0])
                    print(result[0][0])
                    # Save predictions, if requested
                    if bb_attack_config.save and attacker_obj.supports_saving_preds:
                        save_dic = attacker_obj.wrap_preds_to_save(result)

                    # Keep saving results (more I/O, minimal loss of information in crash)
                    logger.save()

    if args.victim_path:
        def joinpath(x, y): return os.path.join(
            args.victim_path, str(x), str(y))
        for i in range(1, 3+1):
            models_1_path = joinpath(data_config.value, i)
            model_2_paths = [joinpath(v, i) for v in attack_config.values]
            single_evaluation(models_1_path, model_2_paths)
    else:
        single_evaluation()
