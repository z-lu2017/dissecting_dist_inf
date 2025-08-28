import numpy as np
from typing import Tuple
from typing import List, Callable
from scipy.stats import gaussian_kde

from distribution_inference.attacks.blackbox.core import Attack, threshold_test_per_dist, PredictionsOnDistributions
from sklearn.metrics import mean_squared_error

def pinball_loss(errors: np.ndarray, tau: float) -> np.ndarray:
    return np.where(errors >= 0, tau * errors, (tau - 1) * errors)

class LossAndThresholdAttack(Attack):
    def attack(self,
                preds_adv: PredictionsOnDistributions,
                preds_vic: PredictionsOnDistributions,
                ground_truth: Tuple[List, List] = None,
                calc_acc: Callable = None,
                epochwise_version: bool = False,
                not_using_logits: bool = False,
                regression_task: bool = False):
    
        """
            Perform Threshold-Test and Loss-Test attacks, optionally using pinball loss.
        """
        # TODO: fix ... was trying to use an asymmetric metric to acccount for
        # under/over predictions but did not get to it/results did not seem to improve
        
        regression_task_metric = "mse"
        
        # if regression_task_metric=="asymmetric" and not hasattr(self, '_pinball_tuned'):
        #     y1 = np.array(ground_truth[0]).flatten()
        #     errs11 = np.array(preds_adv.preds_on_distr_1.preds_property_1) - y1[None, :]
        #     errs21 = np.array(preds_adv.preds_on_distr_1.preds_property_2) - y1[None, :]
        #     mean_err1 = np.concatenate([errs11, errs21]).mean()
        #     y2 = np.array(ground_truth[1]).flatten()
        #     errs12 = np.array(preds_adv.preds_on_distr_2.preds_property_1) - y2[None, :]
        #     errs22 = np.array(preds_adv.preds_on_distr_2.preds_property_2) - y2[None, :]
        #     mean_err2 = np.concatenate([errs12, errs22]).mean()

        #     def choose_tau(mean_err: float) -> float:
        #         if mean_err > 0:
        #             return 0.4  # adv overestimates
        #         elif mean_err < 0:
        #             return 0.6  # adv underestimates
        #         else:
        #             return 0.5

        #     self._tau_for_split1 = choose_tau(mean_err1)
        #     self._tau_for_split2 = choose_tau(mean_err2)
        #     self._pinball_tuned = True
        # else:
        #     self._tau_for_split1 = None
        #     self._tau_for_split2 = None

        # assert calc_acc is not None, "Must provide function to compute accuracy"
        # assert ground_truth is not None, "Must provide ground truth to compute accuracy"
        # assert not (
        #     self.config.multi2 and self.config.multi), "No implementation for both multi model"
        # assert not (
        #     epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"

        if regression_task_metric == "asymmetric":
            def calc_acc(data, labels, multi_class = False):
                n_models = data.shape[1]
                tau = self._tau_for_split1 if data.shape[0] == preds_adv.preds_on_distr_1.preds_property_1.shape[0] else self._tau_for_split2
                pinball_scores = np.zeros(n_models)
                for i in range(n_models):
                    y_pred = data[:, i]
                    residual = labels - y_pred
                    loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual)
                    pinball_scores[i] = np.mean(loss)
                return pinball_scores
        elif regression_task_metric == "mse": 
            def calc_acc(data, labels, multi_class=False):
                n_models = data.shape[1]
                mse_scores = np.zeros(n_models)
                for i in range(n_models):
                    y_pred = data[:, i]
                    # TEMPORARY FIX - REVISIT
                    y_pred = np.nan_to_num(y_pred, nan=0.0)
                    labels = np.nan_to_num(labels, nan=0.0)
                    mse_scores[i] = mean_squared_error(labels, y_pred)
                return mse_scores
        elif regression_task_metric == "KL_kde": 
            def KL_kde(
                p: np.ndarray,
                q: np.ndarray,
                n_points: int = 500,
                plot: bool = False
            ):
                min_val = min(np.min(p), np.min(q))
                max_val = max(np.max(p), np.max(q))
            
                support = np.linspace(min_val, max_val, n_points)
                
                p_kde = gaussian_kde(p)
                q_kde = gaussian_kde(q)
                
                p_vals = p_kde(support) + 1e-8
                q_vals = q_kde(support) + 1e-8

                if plot: 
                    plot_KL_kde(p_vals, q_vals, support)

                # bin width/step size used to approximate area utc
                dx = support[1] - support[0]
                p_vals /= np.sum(p_vals) * dx
                q_vals /= np.sum(q_vals) * dx

                kl = np.sum(p_vals * np.log(p_vals / q_vals)) * dx

                return kl
            def calc_acc(data, labels, multi_class=False):
                n_models = data.shape[1]
                kl_scores = np.zeros(n_models)
                for i in range(n_models):
                    y_pred = data[:, i]
                    kl_scores[i] = KL_kde(y_pred, labels)
                return kl_scores
        else:
            assert calc_acc is not None, "Must provide function to compute accuracy or enable pinball/regression"
        
        # Get accuracies on first data distribution using prediction from shadow/victim models
        adv_accs_1, victim_accs_1, acc_1 = threshold_test_per_dist(
            calc_acc,
            preds_adv.preds_on_distr_1,
            preds_vic.preds_on_distr_1,
            ground_truth[0],
            self.config,
            epochwise_version=epochwise_version)
        # Get accuracies on second data distribution
        adv_accs_2, victim_accs_2, acc_2 = threshold_test_per_dist(
            calc_acc,
            preds_adv.preds_on_distr_2,
            preds_vic.preds_on_distr_2,
            ground_truth[1],
            self.config,
            epochwise_version=epochwise_version)
        # adv_acc: how often the theshold rule correctly distinguishes between the 
        # adv two different prediction sets (assume victim can't really be better than this)

        # Get best adv accuracies for both distributions, across all ratios
        chosen_distribution = 0
        if np.max(adv_accs_1) > np.max(adv_accs_2):
            adv_accs_use, victim_accs_use = adv_accs_1, victim_accs_1
        else:
            adv_accs_use, victim_accs_use = adv_accs_2, victim_accs_2
            chosen_distribution = 1

        # Of the chosen distribution, pick the one with the best accuracy
        # out of all given ratios
        chosen_ratio_index = np.argmax(adv_accs_use)
        if epochwise_version:
            victim_acc_use = [x[chosen_ratio_index] for x in victim_accs_use]
        else:
            victim_acc_use = victim_accs_use[chosen_ratio_index]
        # victim_accuracy: how often the same adv theshold correctly distinguishes
        # between the victim's model predictions

        # Loss test
        if epochwise_version:
            basic = []
            for i in range(acc_1[0].shape[0]):
                basic.append(
                    self._loss_test(
                        (acc_1[0][i], acc_1[1][i]), # victim a0 on a0 and a1 on a0
                        (acc_2[0][i], acc_2[1][i])  # victim a0 on a1 and a1 on a1
                    )
                )
            basic_chosen = [x[chosen_ratio_index] for x in basic]
        else:
            if self.config.multi2:
                basic = self._loss_multi(acc_1, acc_2)
            else:
                basic = self._loss_test(acc_1, acc_2)
            basic_chosen = basic[chosen_ratio_index]

        choice_information = (chosen_distribution, chosen_ratio_index)
        print([[(victim_acc_use, basic_chosen)], [adv_accs_use[chosen_ratio_index]], choice_information])
        return [[(victim_acc_use, basic_chosen)], [adv_accs_use[chosen_ratio_index]], choice_information]

    def _loss_test(self, acc_1, acc_2):
        basic = []
        for r in range(len(self.config.ratios)):
            preds_1 = (acc_1[0][r, :] > acc_2[0][r, :]) # acc of 0 on 0, 0 on 1
            preds_2 = (acc_1[1][r, :] <= acc_2[1][r, :]) # acc of 1 on 0, 1 on 1
            basic.append(100*(np.mean(preds_1) + np.mean(preds_2)) / 2)
        return basic

    def _loss_multi(self, acc_1, acc_2):
        basic = []
        l = acc_1[0].shape[1]
        for r in range(len(self.config.ratios)):
            preds_1 = []
            preds_2 = []
            for i in range(l):
                # Pick 'multi2' random samples
                sampling = np.random.permutation(l)[:self.config.multi2]
                # Equivalent to majority voting on each model's prediction
                preds_1.append(
                    np.mean(acc_1[0][r, sampling] > acc_2[0][r, sampling]) >= 0.5)
                preds_2.append(
                    np.mean(acc_1[1][r, sampling] <= acc_2[1][r, sampling]) >= 0.5)
            basic.append(100*(np.mean(preds_1) + np.mean(preds_2)) / 2)
        return basic
