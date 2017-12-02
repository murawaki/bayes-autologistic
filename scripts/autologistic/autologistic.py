import numpy as np
import random
from collections import Counter
from collections import OrderedDict

class Autologistic:
    """Execute multinomial Autologistic model

    Args:
        x (numpy vector):
            Vectors whose elements denote typological feature values
            of the target languages. If the element value is -1, which is
            regarded as a missing value
        horizontal_graph, vertical_graph (networkx graph):
            Neighbor graphs on horizontal and vertical
        max_value (int):
            Max value of the feature values
            corresponding to the number of feature types
    """
    def __init__(self, x, vertical_graph, horizontal_graph, max_value,
                 distance_weighting = False,
                 norm_sigma = 10.0,
                 gamma_shape = 1.0,
                 gamma_scale = 0.001,
                 use_m = False,
                 use_emp_mean = False,
                 sample_param_weight = 5,
                 init_vh = 0.0001):
        self.__max_value = max_value
        self.__value_list = list(range(self.__max_value+1))
        self.distance_weighting = distance_weighting
        self.use_m = use_m

        self.norm_sigma = norm_sigma
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.use_emp_mean = use_emp_mean

        self.__v_p = init_vh
        self.__h_p = init_vh
        self.__m_p = init_vh if self.use_m else 0.0

        # Initialize u with the distribution probability
        # of each feature value
        value_dist = Counter()
        for i in np.where(x != -1)[0]:
            value_dist[x[i]] += 1
        _sum = sum(value_dist.values())

        self.__u_p = np.log(np.array([(value_dist[i] + 0.01) / _sum for i in range(self.__max_value+1)]))
        self.__u_p = self.__u_p - np.min(self.__u_p)
        self.__u_p0 = self.__u_p.copy()
        value_dist = dict([(i, value_dist[i] / _sum) for i in range(self.__max_value+1)])
        print(value_dist)
        print("initial params",
              self.__v_p,
              self.__h_p,
              self.__m_p, # dummy
              self.__u_p)

        self.__vertical_graph = vertical_graph
        self.__horizontal_graph = horizontal_graph

        self.__x = x.copy()
        self.sample_param_weight = sample_param_weight
        self.tasks = None

    def estimate_with_missing_value(self,
                                    test_data=None,
                                    pretrain_iter=0,
                                    param_iter=200):
        """
        Estimate parameters on incomplete data

        Args:
            test_data (numpy vector):
                Vector for checking the accuracy (hit rate)
                The accuracy (hit rate) is calculated
                by comparing estimated x and this

        Returns:
            tuple: estimated x and parameters

        """
        print('Initialize the missing values')
        missing_indexes = self.__init_missing_values()

        universal_accuracy = OrderedDict({'total': 0, 'correct': 0})
        neighborhood_accuracy = OrderedDict({'total': 0, 'correct': 0})
        if test_data:
            # Accuracy of universal model
            universal_accuracy = self._universal_model(test_data,
                                                       missing_indexes)

            print('Universal accuracy',
                  universal_accuracy['correct'] / universal_accuracy['total'])

            # Accuracy of universal model
            global_majority_accuracy\
                = self._global_majority_model(test_data, missing_indexes)

            print('Global majority accuracy',
                  global_majority_accuracy['correct']
                  / global_majority_accuracy['total'])

            # Accuracy of neighborhodd model
            difference = self.__x[test_data['target_indexes']] - test_data['answers']
            neighborhood_accuracy['correct'] = len(np.where(difference == 0)[0])
            neighborhood_accuracy['total'] = len(difference)
            print('Neighborhood accuracy',
                  neighborhood_accuracy['correct'] / neighborhood_accuracy['total'])

        if pretrain_iter > 0:
            print('Start pretraining')
            self.estimate(
                missing_indexes, test_data,
                only_x=True,
                _iter=pretrain_iter
            )
            self.tasks = None

        print('Start the estimation')

        # Estimate parameters
        self.estimate(
            missing_indexes, test_data,
            _iter=param_iter
        )

        # Reestimate vector x on missing indices with calculated parameters
        change_rate = 0
        ITER_REESTIMATE = 2495 # 2595
        BURN_IN = 0 # 100
        SAMPLING_INTERVAL = 5
        sample_num = (ITER_REESTIMATE-BURN_IN)//SAMPLING_INTERVAL+2
        sampled_xs = np.zeros((sample_num, len(self.__x)), int)
        sampled_xs[0, :] = self.__x.copy()
        sampled_params = []
        previous_x = self.__x.copy()

        for i in range(ITER_REESTIMATE+1):
            self.estimate(
                missing_indexes, test_data,
                _iter=1
            )

            if i >= BURN_IN and i % SAMPLING_INTERVAL == 0:
                previous_idx = (i-BURN_IN) // SAMPLING_INTERVAL
                c_rate = (sampled_xs[previous_idx] != self.__x).sum() / len(self.__x)
                sampled_xs[previous_idx+1, :] = self.__x
                print('Change rate', c_rate)
                change_rate += c_rate
                sampled_params.append({
                    "v": self.__v_p,
                    "h": self.__h_p,
                    "m": self.__m_p,
                    "u": self.__u_p.tolist(),
                })

            previous_x = self.__x.copy()

        # Pickup most occured feature value in the sampling on each language
        sampled_values_dists = {}
        for i in missing_indexes:
            sampled_values_dist = Counter(sampled_xs[1:, i])
            print('Sampled value distribution', i, sampled_values_dist)
            self.__x[i] = sampled_values_dist.most_common(1)[0][0]
            sampled_values_dists[i] = sampled_values_dist

        # Output estimated parameters
        proposed_accuracy = dict({'total': 0, 'correct': 0})
        if test_data:
            proposed_accuracy['total'] = len(test_data['answers'])
            proposed_accuracy['correct']\
                = int((self.__x[test_data['target_indexes']]
                       == test_data['answers']).sum())
            print('Accuracy:',
                  proposed_accuracy['correct'] / proposed_accuracy['total'],
                  len(test_data['answers']))

        v = self.__v_p
        h = self.__h_p
        m = self.__m_p
        u = self.__u_p
        estimated_parameters = OrderedDict()
        estimated_parameters['v'] = np.mean(np.array(list(map(lambda x: x['v'], sampled_params))))
        estimated_parameters['h'] = np.mean(np.array(list(map(lambda x: x['h'], sampled_params))))
        estimated_parameters['m'] = np.mean(np.array(list(map(lambda x: x['h'], sampled_params)))) if self.use_m else 0.0
        estimated_parameters['u'] = np.mean(np.array(list(map(lambda x: np.array(x['u']), sampled_params))), axis=0).tolist()

        print('Reestimated parameters: ', v, h, m, u)

        info_others = OrderedDict()
        info_others['sampled_params'] = sampled_params
        info_others['sampled_values_distributions'] = sampled_values_dists

        if test_data:
            accuracy_info = OrderedDict()
            accuracy_info['proposed'] = proposed_accuracy,
            accuracy_info['universal'] = universal_accuracy,
            accuracy_info['neighborhood'] = neighborhood_accuracy,
            accuracy_info['global_majority'] = global_majority_accuracy
            info_others['accuracies'] = accuracy_info
            return (self.__x,
                    estimated_parameters,
                    info_others)

        return (self.__x, estimated_parameters, info_others)

    def _universal_model(self, test_data, missing_indexes):
        ts = np.zeros(len(test_data['target_indexes']), dtype=int)
        value_dist, target_indices = self._make_value_distribution(test_data, missing_indexes)
        probs = [value_dist[i] for i in self.__value_list]
        probs /= np.sum(probs)

        for i in range(len(ts)):
            ts[i] = np.random.choice(self.__value_list,
                                        p=probs)
        difference = ts - test_data['answers']

        accuracy = OrderedDict()
        accuracy['correct'] = len(np.where(difference == 0)[0])
        accuracy['total'] = len(difference)
        return accuracy

    def _global_majority_model(self, test_data, missing_indexes):
        ts = np.zeros(len(test_data['target_indexes']), dtype=int)
        value_dist, target_indices = self._make_value_distribution(test_data, missing_indexes)
        probs = [value_dist[i] for i in self.__value_list]
        probs /= np.sum(probs)

        majority_feature_value \
            = Counter(self.__x[target_indices]).most_common(1)[0][0]
        ts[:] = majority_feature_value
        difference = ts - test_data['answers']

        accuracy = OrderedDict()
        accuracy['correct'] = len(np.where(difference == 0)[0])
        accuracy['total'] = len(difference)
        return accuracy

    def _make_value_distribution(self, test_data, missing_indexes):
        target_indices = np.ones(len(self.__x), dtype=bool)
        target_indices[missing_indexes] = False
        value_dist = Counter(self.__x[target_indices])
        return value_dist, target_indices

    def estimate(self,
                 missing_indexes=None, test_data=None, only_x=False,
                 _iter=200,
    ):
        """
        Estimate parameters by applying autologistic model

        Args:
            # previous_v_sum, previous_h_sum, previous_u_sum
            #     Previous values of V_{i,j}, H_{i,j}, U_{i,j}
            missing_indexes (np vector):
                ``x``'s indexes of missing values
            test_data (np vector):
                Vector for checking the precision (hit rate)
                The precision (hit rate) is calculated
                by comparing estimated x and this (output to standard output)

        """
        sampled_x = np.zeros(len(self.__x))

        iter_count = 0
        SAMPLE_X = 0
        SAMPLE_V = 1
        SAMPLE_H = 2
        SAMPLE_M = 3
        SAMPLE_U = 4
        if self.tasks is None:
            self.tasks = []
            if not only_x:
                for i in range(self.sample_param_weight):
                    self.tasks.append((SAMPLE_V, None))
                    self.tasks.append((SAMPLE_H, None))
                    if self.use_m:
                        self.tasks.append((SAMPLE_M, None))
                # sample U_k only once per iteration
                for k in range(self.__max_value + 1):
                    self.tasks.append((SAMPLE_U, k))
            for i in missing_indexes:
                self.tasks.append((SAMPLE_X, i))

        while iter_count < _iter:
            iter_count += 1
            c_x = [0, 0]
            c_v = [0, 0]
            c_h = [0, 0]
            c_m = [0, 0]
            c_u = [0, 0]
            random.shuffle(self.tasks)
            for t_type, t_val in self.tasks:
                if t_type == SAMPLE_X:
                    i = t_val
                    oldv = self.__x[i]
                    newv = self.__sample_x(self.__v_p, self.__h_p, self.__m_p, self.__u_p, self.__x, i)
                    self.__x[i] = newv
                    changed = 0 if oldv == newv else 1
                    c_x[changed] += 1
                elif t_type == SAMPLE_V:
                    changed = self.__sample_param('V', None)
                    c_v[changed] += 1
                elif t_type == SAMPLE_H:
                    changed = self.__sample_param('H', None)
                    c_h[changed] += 1
                elif t_type == SAMPLE_M:
                    changed = self.__sample_param('M', None)
                    c_m[changed] += 1
                elif t_type == SAMPLE_U:
                    changed = self.__sample_param('U', t_val)
                    c_u[changed] += 1
                else:
                    raise NotImplementedError                 
            print(iter_count, np.inf,
                  self.__v_p,
                  self.__h_p,
                  self.__m_p,
                  self.__u_p,
                  "x change rate: ",
                  float(c_x[1]) / sum(c_x),
                  "u change rate: ",
                  (float(c_u[1]) / sum(c_u)) if sum(c_u) > 0 else np.inf,
            )

            if test_data is None:
                continue
            accuracy = (self.__x[test_data['target_indexes']] == test_data['answers']).sum() / len(test_data['answers'])
            print('Accuracy:', accuracy)

    def __neighbor_sum(self, vector, neighbor_graph, target_indexes=None):
        concordants = 0
        if neighbor_graph == self.__horizontal_graph and self.distance_weighting:
            for i, a in enumerate(vector):
                hn = np.array(neighbor_graph.neighbors(i))
                if hn.size <= 0:
                    continue
                for j in hn[vector[hn] == a]:
                    if j > i:
                        concordants += neighbor_graph[i][j]['weight']
        else:
            if self.use_m:
                if neighbor_graph == self.__vertical_graph:
                    is_main, net1, net2 = True, self.__vertical_graph, self.__horizontal_graph
                elif neighbor_graph == self.__horizontal_graph:
                    is_main, net1, net2 = True, self.__horizontal_graph, self.__vertical_graph
                else:
                    is_main, net1, net2 = False, self.__vertical_graph, self.__horizontal_graph
                for i, a in enumerate(vector):
                    l1 = np.array(net1.neighbors(i))
                    l2 = np.array(net2.neighbors(i))
                    if is_main:
                        hn = l1[np.in1d(l1, l2, invert=True)]
                    else:
                        hn = np.intersect1d(l1, l2, assume_unique=True)
                    if hn.size <= 0:
                        continue
                    hn = hn[vector[hn] == a]
                    concordants += (hn > i).sum()                    
            else:
                for i, a in enumerate(vector):
                    hn = np.array(neighbor_graph.neighbors(i))
                    if hn.size <= 0:
                        continue
                    hn = hn[vector[hn] == a]
                    concordants += (hn > i).sum()
        return concordants

    def __sum_vector(self, vector, value):
        return (vector == value).sum()

    def __sample_x(self, v_p, h_p, m_p, u_p, x, i):
        V = np.bincount(x[self.__vertical_graph.neighbors(i)], minlength=self.__max_value+1)
        if self.distance_weighting:
            H = np.zeros(self.__max_value+1)
            hn = self.__horizontal_graph[i]
            for j, attr in hn.items():
                H[x[j]] += attr['weight']
        else:
            H = np.bincount(x[self.__horizontal_graph.neighbors(i)], minlength=self.__max_value+1)
        if self.use_m:
            # TODO: this is too slow. Construct three neighbor graphs at initialization and use them
            mn = np.intersect1d(self.__vertical_graph.neighbors(i), self.__horizontal_graph.neighbors(i), assume_unique=True).astype(np.int32)
            M = np.bincount(x[mn], minlength=self.__max_value+1)
            probs = v_p * (V - M) + h_p * (H - M) + m_p * M + u_p
        else:
            probs = v_p * V + h_p * H + u_p
        probs = np.exp(probs - np.max(probs))
        probs = probs / probs.sum()
        return np.random.choice(self.__value_list, p=probs)

    def __sample_param(self, t_type, idx):
        logr = 0.0
        if t_type == 'U':
            oldval = self.__u_p[idx]
            newval = np.random.normal(loc=oldval, scale=0.01)
            # P(theta') / P(theta)
            mean = self.__u_p0[idx] if self.use_emp_mean else 0.0
            logr += ((oldval - mean) ** 2 - (newval - mean) ** 2) / (2.0 * self.norm_sigma * self.norm_sigma)
            # skip: q(theta|theta', x) / q(theta'|theta, x) for symmetric proposal
            u = self.__u_p.copy()
            u[idx] = newval
            v, h, m = self.__v_p, self.__h_p, self.__m_p
        else:
            if t_type == 'V':
                oldval = self.__v_p
            elif t_type == 'H':
                oldval = self.__h_p
            else:
                oldval = self.__m_p
            P_SIGMA = 0.5
            rate = np.random.lognormal(mean=0.0, sigma=P_SIGMA)
            irate = 1.0 / rate
            newval = rate * oldval
            lograte = np.log(rate)
            logirate = np.log(irate)
            # P(theta') / P(theta)
            # logr += gamma.logpdf(newval, self.gamma_shape, scale=self.gamma_scale) \
            #         - gamma.logpdf(oldval, self.gamma_shape, scale=self.gamma_scale)
            logr += (self.gamma_shape - 1.0) * np.log(rate) - (newval - oldval) / self.gamma_scale
            # q(theta|theta', x) / q(theta'|theta, x)
            logr += (lograte * lograte - logirate * logirate) / (2.0 * P_SIGMA * P_SIGMA) + lograte - logirate
            if t_type == 'V':
                v, h, m, u = newval, self.__h_p, self.__m_p, self.__u_p
                net = self.__vertical_graph
            elif t_type == 'H':
                v, h, m, u = self.__v_p, newval, self.__m_p, self.__u_p
                net = self.__horizontal_graph
            elif t_type == 'M':
                v, h, m, u = self.__v_p, self.__h_p, newval, self.__u_p
                net = None
        llist = list(range(len(self.__x)))
        np.random.shuffle(llist)
        x = self.__x.copy()
        for i in llist:
            x[i] = self.__sample_x(v, h, m, u, x, i)
        if t_type == 'U':
            logr += (oldval - newval) * (self.__sum_vector(x, idx) - self.__sum_vector(self.__x, idx))
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                self.__u_p[idx] = newval
                print("U{}\t{}\t{}\taccepted".format(idx, oldval, newval))
                return True
            else:
                print("U{}\t{}\t{}\trejected".format(idx, oldval, newval))
                return False
        else:
            oldsum = self.__neighbor_sum(self.__x, net)
            newsum = self.__neighbor_sum(x, net)
            logr += (oldval - newval) * (newsum - oldsum)
            if logr >= 0 or np.log(np.random.rand()) < logr:
                # accept
                if t_type == 'V':
                    self.__v_p = newval
                elif t_type == 'H':
                    self.__h_p = newval
                else:
                    self.__m_p = newval
                print("{}\t{}\t{}\taccepted".format(t_type, oldval, newval))
                return True
            else:
                print("{}\t{}\t{}\trejected".format(t_type, oldval, newval))
                return False

    def __non_missing_probs(self, graph, i, missing_indexes):
        target_indexes = list(set(graph.neighbors(i)) - set(missing_indexes))
        if len(target_indexes) == 0:
            return None

        value_dist = Counter(self.__x[target_indexes])
        values = list(value_dist.keys())
        probs = np.array(list(value_dist.values()))
        return (probs / np.sum(probs), values)

    def __init_missing_values(self):
        value_dist = Counter()
        for i in np.where(self.__x != -1)[0]:
            value_dist[self.__x[i]] += 1
        values = list(value_dist.keys())
        init_probs = np.array(list(value_dist.values()))
        init_probs = init_probs / np.sum(init_probs)

        missing_indexes = np.where(self.__x == -1)[0]

        for i in missing_indexes:
            vertical_probs = self.__non_missing_probs(
                self.__vertical_graph, i, missing_indexes)
            horizontal_probs = self.__non_missing_probs(
                self.__horizontal_graph, i, missing_indexes)

            if vertical_probs is None and horizontal_probs is None:
                self.__x[i] = np.random.choice(values, p=init_probs)
            elif horizontal_probs is None:
                self.__x[i] = np.random.choice(vertical_probs[1],
                                                  p=vertical_probs[0])
            elif vertical_probs is None:
                self.__x[i] = np.random.choice(horizontal_probs[1],
                                                  p=horizontal_probs[0])
            else:
                vertical_sampled = np.random.choice(vertical_probs[1],
                                                       p=vertical_probs[0])
                horizontal_sampled = np.random.choice(horizontal_probs[1],
                                                         p=horizontal_probs[0])
                sampled_values = [vertical_sampled, horizontal_sampled]
                self.__x[i] = np.random.choice(sampled_values, p=[0.5, 0.5])
        return missing_indexes
