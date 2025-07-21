import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistribution,PredictionsOnOneDistribution
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


class KLAttack(Attack):
    def attack(self,
                preds_adv: PredictionsOnDistributions,
                preds_vic: PredictionsOnDistributions,
                ground_truth: Tuple[List, List] = None,
                calc_acc: Callable = None,
                epochwise_version: bool = False,
                not_using_logits: bool = False,
                regression_task: bool = False):
        self.not_using_logits = not_using_logits
        self.regression_task = regression_task

        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        # confirm we are entering not epochwise version 
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        if not epochwise_version:
            return self.attack_not_epoch(preds_adv,preds_vic,ground_truth,calc_acc)
        else:
            preds_v = [PredictionsOnDistributions(
                PredictionsOnOneDistribution(preds_vic.preds_on_distr_1.preds_property_1[i],preds_vic.preds_on_distr_1.preds_property_2[i]),
                PredictionsOnOneDistribution(preds_vic.preds_on_distr_2.preds_property_1[i],preds_vic.preds_on_distr_2.preds_property_2[i])
            ) for i in range(len(preds_vic.preds_on_distr_2.preds_property_1))]
            accs,preds=[],[]
            for x in preds_v:
                result = self.attack_not_epoch(preds_adv,x,ground_truth,calc_acc)
                accs.append(result[0][0])
                preds.append(result[0][1])
            return [(accs, preds), (None, None), (None,None)]

    def attack_not_epoch(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None):
        
        # Get values using data from first distribution
        """
            creates sets
            1. adv alpha0 pred on alpha0
            2. adv alpha1 on alpha0
            3. vic alpha0 pred on alpha0
            4. vic alpha1 on alpha0

            1. adv alpha0 pred on alpha1
            2. adv alpha1 on alpha1
            3. vic alpha0 pred on alpha1
            4. vic alpha1 on alpha1
            ...
            computes kl for preds of 
            - 1&3
            - 2&3 
            - 1&4
            - 2&4
            ...
            then does pairwise comp for 
                1&3 - 2&3
                1&4 - 2&4
            ...
            In both cases we expect
                1&3 - 2&3 to be negative
                1&4 - 2&4 to be positive
        """
        preds_1_first, preds_1_second = self._get_kl_preds(
            preds_adv.preds_on_distr_1.preds_property_1,
            preds_adv.preds_on_distr_1.preds_property_2,
            preds_vic.preds_on_distr_1.preds_property_1,
            preds_vic.preds_on_distr_1.preds_property_2)
        # Get values using data from second distribution
        preds_2_first, preds_2_second = self._get_kl_preds(
            preds_adv.preds_on_distr_2.preds_property_1,
            preds_adv.preds_on_distr_2.preds_property_2,
            preds_vic.preds_on_distr_2.preds_property_1,
            preds_vic.preds_on_distr_2.preds_property_2)

        # Note: KL‐voting aggregates Boolean >0 votes per record, which 
        # requires a (n_records × n_pairs) layout.
        # In the regression branch we flatten all pairwise KDE differences 
        # into a single vector, so theres no per record grouping to vote 
        # over ... KL‐voting isn’t meaningful here

        if self.regression_task: 
            preds_first = np.concatenate((preds_1_first, preds_2_first)) 
            preds_second = np.concatenate((preds_1_second, preds_2_second)) 
            
            preds = np.concatenate((preds_first, preds_second))
            
            decision_boundry = 0.0
        else: 
            preds_first = np.concatenate((preds_1_first, preds_2_first), 1) 
            preds_second = np.concatenate((preds_1_second, preds_2_second), 1) 
        
            preds = np.concatenate((preds_first, preds_second))
            if not self.config.kl_voting:
                preds -= np.min(preds, 0)
                preds /= np.max(preds, 0)
            
            preds = np.mean(preds, 1)
            decision_boundry = 0.5

        gt = np.concatenate((np.zeros(preds_first.shape[0]), np.ones(preds_second.shape[0])))
        decision = 100 * np.mean((preds >= decision_boundry) == gt)

        print(f"Decision Accuracy: {decision}")
        choice_information = (None, None)
        return [(decision, preds), (None, None), choice_information]

    def _get_kl_preds(self, ka, kb, kc1, kc2):
        """
        ka & kb are adv preds, kc1 and kc2 are vic preds
        - take predictions and compute softmax/sigmoid to normalize them and create prob dist
        - if self.config.log_odds_order then the log odds is computed to exagerate differences in the probability distributions
            and the top half of data samples with large differences in values are used to compute KL
        - get all unique pairs of model comps (n choose 2)
        """
        ka_, kb_ = ka, kb
        kc1_, kc2_ = kc1, kc2
        if not self.not_using_logits and not self.regression_task:
            if self.config.multi_class:
                ka_, kb_ = softmax(ka), softmax(kb)
                kc1_, kc2_ = softmax(kc1), softmax(kc2)
            else:
                ka_, kb_ = sigmoid(ka), sigmoid(kb)
                kc1_, kc2_ = sigmoid(kc1), sigmoid(kc2)

        # Use log-odds-ratio to order data and pick only top half
        if self.config.log_odds_order and not self.regression_task:
            small_eps = 1e-4
            log_vals_a = np.log((small_eps + ka_) / (small_eps + 1 - ka_))
            log_vals_b = np.log((small_eps + kb_) / (small_eps + 1 - kb_))
            ordering = np.mean(np.abs(log_vals_a - log_vals_b), 0)
            ordering = np.argsort(ordering)[::-1]
            # Pick only first half
            ordering = ordering[:len(ordering) // 2]
            ka_, kb_ = ka_[:, ordering], kb_[:, ordering]
            kc1_, kc2_ = kc1_[:, ordering], kc2_[:, ordering]

        global_min = min(ka_.min(), kb_.min(), kc1_.min(), kc2_.min())
        global_max = max(ka_.max(), kb_.max(), kc1_.max(), kc2_.max())
        support = np.linspace(global_min, global_max, 500) # 500=point to use TODO: add in config
        dx = support[1] - support[0]

        if self.regression_task:
            KL_est = lambda p, q: KL_kde(p, q, support, dx)
        else:
            KL_est = lambda p, q: KL(p, q, multi_class=self.config.multi_class)

        print("\nComputing KL scores ...")
        KL_vals_1_a = np.array([KL_est(ka_[i], kc1_[i]) for i in range(ka_.shape[0])])
        self._check(KL_vals_1_a)
        KL_vals_1_b = np.array([KL_est(kb_[i], kc1_[i]) for i in range(kb_.shape[0])])
        self._check(KL_vals_1_b)
        KL_vals_2_a = np.array([KL_est(ka_[i], kc2_[i]) for i in range(ka_.shape[0])])
        self._check(KL_vals_2_a)
        KL_vals_2_b = np.array([KL_est(kb_[i], kc2_[i]) for i in range(kb_.shape[0])])
        self._check(KL_vals_2_b)
        print("... done\n")

        # debug_kl_distributions(np.concatenate((KL_vals_1_a, KL_vals_2_a)),
        #                np.concatenate((KL_vals_1_b, KL_vals_2_b)))


        # pairwise compare
        xx, yy = np.triu_indices(ka.shape[0], k=1)
        rnd = np.random.permutation(xx.size)[: int(self.config.kl_frac * xx.size)]
        xx, yy = xx[rnd], yy[rnd]
        preds_first = self._pairwise_compare(KL_vals_1_a[:, None], KL_vals_1_b[:, None], xx, yy)
        preds_second = self._pairwise_compare(KL_vals_2_a[:, None], KL_vals_2_b[:, None], xx, yy)

        print("Split 1 votes >0:", np.mean(preds_first > 0))
        print("Split 2 votes >0:", np.mean(preds_second > 0))

        return preds_first, preds_second
    
    def _check(self, x):
        if np.sum(np.isinf(x)) > 0 or np.sum(np.isnan(x)) > 0:
            print("Invalid values:", x)
            raise ValueError("Invalid values found!")

    def _pairwise_compare(self, x, y, xx, yy):
        """
        Computes every single combinatation of comparisons before selecting subset
        """
        if not self.regression_task: 
            x_ = np.expand_dims(x, 2)
            y_ = np.expand_dims(y, 2)
            y_ = np.transpose(y_, (0, 2, 1))
            if self.config.kl_voting and not self.regression_task:
                pairwise_comparisons = (x_ > y_)
            else:
                pairwise_comparisons = (x_ - y_)
            preds = np.array([z[xx, yy] for z in pairwise_comparisons])
        else: 
            y = np.transpose(y, (1, 0))
            pairwise_comparisons = (x - y)
            preds = pairwise_comparisons[xx, yy]
        return preds

def sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)

def softmax(x):
    z = x - np.max(x, -1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, -1, keepdims=True)


def KL(x, y, multi_class: bool = False):
    small_eps = 1e-4
    x_ = np.clip(x, small_eps, 1 - small_eps)
    y_ = np.clip(y, small_eps, 1 - small_eps)
    if multi_class:
        return np.mean(np.sum(x_ * (np.log(x_) - np.log(y_)),axis=2),axis=1)
    else:
        x__, y__ = 1 - x_, 1 - y_
        first_term = x_ * (np.log(x_) - np.log(y_))
        second_term = x__ * (np.log(x__) - np.log(y__))
    return np.mean(first_term + second_term, 1)

def KL_kde(
        p: np.ndarray, 
        q: np.ndarray, 
        support: np.ndarray, 
        dx: float
    ):
    # scott's rule used for bandwidth by default
    p_kde = gaussian_kde(p)
    q_kde = gaussian_kde(q)
    
    p_vals = p_kde(support) + 1e-8
    q_vals = q_kde(support) + 1e-8
    
    # bin width/step size used to approximate area utc
    dx = support[1] - support[0]
    p_vals /= np.sum(p_vals) * dx
    q_vals /= np.sum(q_vals) * dx

    kl = np.sum(p_vals * np.log(p_vals / q_vals)) * dx

    return kl

def debug_kl_distributions(KL_vals_split1, KL_vals_split2):
    # Print summary statistics
    print("Split 1 — mean: {:.4f}, median: {:.4f}, std: {:.4f}".format(
        np.mean(KL_vals_split1), np.median(KL_vals_split1), np.std(KL_vals_split1)))
    print("Split 2 — mean: {:.4f}, median: {:.4f}, std: {:.4f}".format(
        np.mean(KL_vals_split2), np.median(KL_vals_split2), np.std(KL_vals_split2)))
    
    plt.figure(figsize=(8, 4))
    plt.hist(KL_vals_split1, bins=30, alpha=0.5, label="Split 1 KL")
    plt.hist(KL_vals_split2, bins=30, alpha=0.5, label="Split 2 KL")
    plt.xlabel("KL divergence")
    plt.ylabel("Frequency")
    plt.title("KL Distributions by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/ujx4ab/ondemand/dissecting_dist_inf/debug_KL.png')