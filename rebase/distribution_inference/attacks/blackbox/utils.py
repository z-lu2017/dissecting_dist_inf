from collections import OrderedDict
from typing import List, Tuple
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch as ch
import gc

from distribution_inference.attacks.blackbox.per_point import PerPointThresholdAttack
from distribution_inference.attacks.blackbox.standard import LossAndThresholdAttack
from distribution_inference.attacks.blackbox.core import PredictionsOnOneDistribution
from distribution_inference.datasets.base import CustomDatasetWrapper
from distribution_inference.attacks.blackbox.epoch_loss import Epoch_LossAttack
from distribution_inference.attacks.blackbox.epoch_threshold import Epoch_ThresholdAttack
from distribution_inference.attacks.blackbox.epoch_perpoint import Epoch_Perpoint
from distribution_inference.attacks.blackbox.epoch_meta import Epoch_Tree
from distribution_inference.attacks.blackbox.perpoint_choose import PerPointChooseAttack
from distribution_inference.attacks.blackbox.perpoint_choose_dif import PerPointChooseDifAttack
from distribution_inference.attacks.blackbox.KL import KLAttack
from distribution_inference.attacks.blackbox.generative import GenerativeAttack
from distribution_inference.attacks.blackbox.binary_perpoint import BinaryPerPointThresholdAttack
#from distribution_inference.attacks.blackbox.KL_regression import KLRegression
#from distribution_inference.attacks.blackbox.label_KL import label_only_KLAttack
#from distribution_inference.attacks.blackbox.zhang import ZhangAttack

ATTACK_MAPPING = {
    "threshold_perpoint": PerPointThresholdAttack,
    "loss_and_threshold": LossAndThresholdAttack,
    "single_update_loss": Epoch_LossAttack,
    "single_update_threshold": Epoch_ThresholdAttack,
    "single_update_perpoint": Epoch_Perpoint,
    "epoch_meta": Epoch_Tree,
    "perpoint_choose": PerPointChooseAttack,
    "perpoint_choose_dif": PerPointChooseDifAttack,
    "KL": KLAttack,
    "generative":GenerativeAttack,
    "binary_perpoint": BinaryPerPointThresholdAttack,
    #"KL_regression": KLRegression,
    #"label_KL": label_only_KLAttack,
    #"zhang": ZhangAttack
}


def get_attack(attack_name: str):
    wrapper = ATTACK_MAPPING.get(attack_name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_name} not implemented")
    return wrapper


def calculate_accuracies(data, labels,
                         use_logit: bool = True,
                         multi_class: bool = False):
    """
        Function to compute model-wise average-accuracy on
        given data.
    """
    # Get predictions from each model (each model outputs logits)
    if multi_class:
        assert len(data.shape) == 3, "Multi-class data must be 3D"
        preds = np.argmax(data, axis=2).astype('int')
    else:
        assert len(data.shape) == 2, "Data should be 2D"
        if use_logit:
            preds = (data >= 0).astype('int')
        else:
            preds = (data >= 0.5).astype('int')

    # Repeat ground-truth (across models)
    expanded_gt = np.repeat(np.expand_dims(
        labels, axis=1), preds.shape[1], axis=1)

    return np.average(1. * (preds == expanded_gt), axis=0)


def get_graph_preds(ds, indices,
                    models: List[nn.Module],
                    verbose: bool = False,
                    multi_class: bool = False,
                    latent: int = None):
    """
        Get predictions for given graph models
    """
    X = ds.get_features()
    Y = ds.get_labels()

    predictions = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Generating Predictions")
    for model in iterator:
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        # Get model outputs/preds
        prediction = model(ds.g, X, latent=latent)[indices].detach().cpu().numpy()
        if latent != None and not multi_class:
            prediction = prediction[:, 0]
        predictions.append(prediction)
    
        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()

    predictions = np.stack(predictions, 0)
    gc.collect()
    ch.cuda.empty_cache()

    labels = Y[indices].cpu().numpy()[:, 0]
    return predictions, labels


def get_preds(loader, models: List[nn.Module],
                preload: bool = False,
                verbose: bool = True,
                multi_class: bool = False,
                latent: int = None,
                return_props_labels: bool = False,
                use_hidden_state: bool = False,
                use_cell_state: bool = False
            ):
    """
        Get predictions for given models on given data
    """
    # Check if models are graph-related
    # TEMPORARY FIX - REVISIT
    if models.is_graph_model:
        return get_graph_preds(ds=loader[0],
                               indices=loader[1],
                               models=models,
                               verbose=verbose,
                               latent=latent,
                               multi_class=multi_class)

    predictions = []
    ground_truth = []
    inputs = []

    prop_labels = [] if return_props_labels else None

    for data in loader:
        if len(data) == 2:
            features, labels = data
        else:
            features, labels, property_labels = data

        ground_truth.append(labels.cpu().numpy())
        if return_props_labels:
            prop_labels.append(property_labels.cpu().numpy())
        if preload:
            inputs.append(features.cuda())

    ground_truth = np.concatenate(ground_truth, axis=0)
    if return_props_labels:
        prop_labels = np.concatenate(prop_labels, axis=0)

    # Get predictions for each model
    iterator = models
    # TEMPORARY FIX - REVISIT
    #if verbose:
    #    iterator = tqdm(iterator, desc="Generating Predictions")
    #for model in iterator:
    model = iterator
   # model = model.cuda()
    model.eval()
    ch.cuda.empty_cache()

    with ch.no_grad():
        predictions_on_model = []

        if preload:
            for data_batch in inputs:
                if latent != None:
                    prediction = model(data_batch, latent=latent).detach()
                else:
                    # TODO: figure out use hidden for other logic branches
                    if use_hidden_state: 
                        prediction = model(data_batch, return_hidden=use_hidden_state).detach()
                        prediction = prediction.squeeze()
                    elif use_cell_state:
                        prediction = model(data_batch, return_cell=use_cell_state).detach()
                        prediction = prediction.squeeze()
                    else: 
                        prediction = model(data_batch).detach()

                        # If None for whatever reason, re-run
                        # Weird bug that pops in every now and then
                        # Was valid only for LR in Sklearn models- commenting out for now
                        # if prediction is None:
                        #     if latent != None:
                        #         prediction = model(data_batch, latent=latent).detach()
                        #     else:
                        #         prediction = model(data_batch).detach()

                    if not multi_class:
                        prediction = prediction[:, 0]
                predictions_on_model.append(prediction.cpu())
        else:
                # Iterate through data-loader
            for data in loader:
                data_points, labels, _ = data
                if latent != None:
                    prediction = model(data_points, latent=latent).detach()
                else:
                    #print("input size -1 = ", data_points.size())
                    #time.sleep(100)
                    prediction = model(data_points).detach()
                    if not multi_class:
                        prediction = prediction[:, 0]
                predictions_on_model.append(prediction)

        predictions_on_model = ch.cat(predictions_on_model).cpu().numpy()
        predictions.append(predictions_on_model)
        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()
    predictions = np.stack(predictions, 0)
    if preload:
        del inputs
    gc.collect()
    ch.cuda.empty_cache()

    return predictions, ground_truth, prop_labels


def _get_preds_accross_epoch(models,
                             loader,
                             preload: bool = False,
                             multi_class: bool = False):
    preds = []
    for e in models:
        p, gt = get_preds(loader, e, preload, multi_class=multi_class)
        preds.append(p)

    return (np.array(preds), np.array(gt))


def get_preds_epoch_on_dis(
        models,
        loader,
        preload: bool = False,
        multi_class: bool = False):
    preds1, gt = _get_preds_accross_epoch(
        models[0], loader, preload, multi_class)
    preds2, _ = _get_preds_accross_epoch(
        models[1], loader, preload, multi_class)
    preds_wrapped = [PredictionsOnOneDistribution(
        preds_property_1=p1,
        preds_property_2=p2
    ) for p1, p2 in zip(preds1, preds2)]
    return (preds_wrapped, gt)


def _get_preds_for_vic_and_adv(
        models_vic: List[nn.Module],
        models_adv: List[nn.Module],
        loader,
        epochwise_version: bool = False,
        preload: bool = False,
        multi_class: bool = False,
        return_props_labels: bool = False,
        regression_task: bool = False,
        use_hidden_state: bool = False, 
        use_cell_state: bool = False
        ):
            
    return_props_labels = True
    # Sklearn models do not support logits- take care of that
    # TEMPORARY FIX - REVISIT
    use_prob_adv = models_adv.is_sklearn_model
    use_prob_vic = models_vic.is_sklearn_model
    '''
    #use_prob_adv = models_adv[0].is_sklearn_model
    if epochwise_version:
        use_prob_vic = models_vic[0][0].is_sklearn_model
    else:
        use_prob_vic = models_vic[0].is_sklearn_model
    not_using_logits = use_prob_adv or use_prob_vic
    '''
    not_using_logits = use_prob_adv or use_prob_vic

    if type(loader) == tuple:
        # Same data is processed differently for vic/adv
        loader_vic, loader_adv = loader
    else:
        # Same loader
        loader_adv = loader
        loader_vic = loader
    
    def to_preds(x):
        exp = np.exp(x)
        return exp / (1 + exp)

    # check loader
    prop_labels = None
    # Get predictions for adversary models and data
    preds_adv, ground_truth_repeat, prop_labels = get_preds(
        loader_adv, models_adv, preload=preload,
        multi_class=multi_class, return_props_labels=return_props_labels,
        use_hidden_state=use_hidden_state, use_cell_state=use_cell_state)
    if not regression_task and not_using_logits and not use_prob_adv:
        preds_adv = to_preds(preds_adv)

    # Get predictions for victim models and data
    if epochwise_version:
        # Track predictions for each epoch
        preds_vic = []
        # TEMPORARY FIX
        #for models_inside_vic in tqdm(models_vic):
            # TODO: fix this to handle return prop lables. atm regression task will never do this
            # TODO: fix this to handle hidden state/cell state, same as above
        models_inside_vic = models_vic
        preds_vic_inside, ground_truth, _ = get_preds(
                loader_vic, models_inside_vic, preload=preload,
                verbose=False, multi_class=multi_class)
        if not regression_task and not_using_logits and not use_prob_vic:
            preds_vic_inside = to_preds(preds_vic_inside)

            # In epoch-wise mode, we need prediction results
            # across epochs, not models
        preds_vic.append(preds_vic_inside)
    else:
        preds_vic, ground_truth, _ = get_preds(
            loader_vic, models_vic, preload=preload,
            multi_class=multi_class, use_hidden_state=use_hidden_state,
            use_cell_state=use_cell_state)
    
    #assert np.all(ground_truth == ground_truth_repeat), "Val loader is shuffling data!"
    return preds_vic, preds_adv, ground_truth, not_using_logits, prop_labels


def get_vic_adv_preds_on_distr_seed(
        models_vic: Tuple[List[nn.Module], List[nn.Module]],
        models_adv: Tuple[List[nn.Module], List[nn.Module]],
        loader,
        epochwise_version: bool = False,
        preload: bool = False,
        multi_class: bool = False):

    preds_vic_1, preds_adv_1, ground_truth = _get_preds_for_vic_and_adv(
        models_vic[0], models_adv[0], loader,
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class)
    # Get predictions for second set of models
    preds_vic_2, preds_adv_2, _ = _get_preds_for_vic_and_adv(
        models_vic[1], models_adv[1], loader,
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class)
    adv_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_adv_1,
        preds_property_2=preds_adv_2
    )
    vic_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_vic_1,
        preds_property_2=preds_vic_2
    )
    return (adv_preds, vic_preds, ground_truth)


"""
Might be an error here ... says that ds_obj belongs to vic, but passing in adv in blackbox
"""
def get_vic_adv_preds_on_distr(
        models_vic: Tuple[List[nn.Module], List[nn.Module]],
        models_adv: Tuple[List[nn.Module], List[nn.Module]],
        ds_obj: CustomDatasetWrapper,
        batch_size: int,
        epochwise_version: bool = False,
        preload: bool = False,
        multi_class: bool = False,
        make_processed_version: bool = False,
        return_props_labels: bool = False,
        regression_task: bool = False, 
        use_hidden_state: bool = False,
        use_cell_state: bool = False
        ):

    # Check if models are graph-related
    are_graph_models = False
    '''
    if epochwise_version:
        if models_vic[0][0][0].is_graph_model:
            are_graph_models = True
    else:
        if models_vic[0][0].is_graph_model:
            are_graph_models = True
    '''
    
    if are_graph_models:
        # No concept of 'processed'
        data_ds, (_, test_idx) = ds_obj.get_loaders(batch_size=batch_size)
        loader_vic = (data_ds, test_idx)
        loader_adv = loader_vic
    else:
        loader_for_shape, loader_vic = ds_obj.get_loaders(batch_size=batch_size)
        # adv_datum_shape = next(iter(loader_for_shape))[0].shape[1:]
        if make_processed_version:
            # Make version of DS for victim that processes data
            # before passing on
            adv_datum_shape = ds_obj.prepare_processed_data(loader_vic)
            loader_adv = ds_obj.get_processed_val_loader(batch_size=batch_size)
        else:
            loader_adv = loader_vic

    # TODO: Use preload logic here to speed things even more
    # Get predictions for first set of models
    preds_vic_1, preds_adv_1, ground_truth, not_using_logits, prop_labels = _get_preds_for_vic_and_adv(
        models_vic[0], models_adv[0],
        (loader_vic, loader_adv),  # both loaders are from same adversarial distribution
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class,
        return_props_labels=return_props_labels,
        regression_task=regression_task,
        use_hidden_state=use_hidden_state,
        use_cell_state=use_cell_state)
    # Get predictions for second set of models
    if models_vic[1] is not None and models_adv[1] is not None:
        preds_vic_2, preds_adv_2, _, _, _ = _get_preds_for_vic_and_adv(
            models_vic[1], models_adv[1],
            (loader_vic, loader_adv),
            epochwise_version=epochwise_version,
            preload=preload,
            multi_class=multi_class,
            return_props_labels=return_props_labels,
            regression_task=regression_task,
            use_hidden_state=use_hidden_state,
            use_cell_state=use_cell_state)

        adv_preds = PredictionsOnOneDistribution(
            preds_property_1=preds_adv_1,
            preds_property_2=preds_adv_2
        )
        vic_preds = PredictionsOnOneDistribution(
            preds_property_1=preds_vic_1,
            preds_property_2=preds_vic_2
        )
        return adv_preds, vic_preds, ground_truth, not_using_logits, prop_labels
    
    else: 
        print("Only getting prediction for 1 Adv and 1 Victim Model")
        return preds_adv_1, preds_vic_1, ground_truth, not_using_logits, prop_labels


def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups):
    """ Compute the key metrics """
    from aif360.metrics import ClassificationMetric
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * \
        (classified_metric_pred.true_positive_rate() +
         classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = \
        classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = \
        classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = \
        classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    metrics["False discovery rate difference"] = \
        classified_metric_pred.false_discovery_rate_difference()
    metrics["False discovery rate ratio"] = \
        classified_metric_pred.false_discovery_rate_ratio()
    metrics["False omission rate difference"] = \
        classified_metric_pred.false_omission_rate_difference()
    metrics["False omission rate ratio"] = \
        classified_metric_pred.false_omission_rate_ratio()
    metrics["False negative rate difference"] = \
        classified_metric_pred.false_negative_rate_difference()
    metrics["False negative rate ratio"] = \
        classified_metric_pred.false_negative_rate_ratio()
    metrics["False positive rate difference"] = \
        classified_metric_pred.false_positive_rate_difference()
    metrics["False positive rate ratio"] = \
        classified_metric_pred.false_positive_rate_ratio()

    return metrics
