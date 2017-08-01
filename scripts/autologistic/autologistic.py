import numpy as np
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
                 with_lasso=False, use_softplus=False, u_lambda=0.0):
        self.__max_value = max_value
        self.__value_list = list(range(self.__max_value+1))
        self.__with_lasso = with_lasso
        self.__use_softplus = use_softplus
        self.__u_lambda = u_lambda # L2 regularization

        print("model settings:\tsoftplus: {}\tu_lambda: {}".format(self.__use_softplus, self.__u_lambda))

        init_value = -5.0 if self.__use_softplus else 0.0
        self.__v_p = init_value  # [random.uniform(-5.0, 0.5)])
        self.__h_p = init_value  # [random.uniform(-5.0, 0.5)])

        # Initialize u with the distribution probability
        # of each feature value
        value_dist = Counter()
        for i in np.where(x != -1)[0]:
            value_dist[x[i]] += 1
        _sum = sum(value_dist.values())

        self.__u_p = np.log(np.array([(value_dist[i] + 0.01) / _sum for i in range(self.__max_value+1)]))
        self.__u_p0 = self.__u_p.copy()
        value_dist = dict([(i, value_dist[i] / _sum) for i in range(self.__max_value+1)])
        print(value_dist)
        # fudge = 1.0e-8
        # self.__u_p = np.array([init_value for i in range(self.__max_value+1)])
        # self.softplus(np.array([np.log(value_dist[i]/10) for i in range(self.__max_value+1)]))
        print("initial params",
              self.softplus(self.__v_p),
              self.softplus(self.__h_p),
              self.__u_p) # self.softplus(self.__u_p))

        self.__vertical_graph = vertical_graph
        self.__horizontal_graph = horizontal_graph

        self.__x = x.copy()

    def estimate_with_missing_value(self, test_data=None):
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

        # ``v_sum'' indicates neighbor_sum
        previous_v_sum = self.__neighbor_sum(self.__x,
                                             self.__horizontal_graph)
        # ``h_sum'' indicates H_{i}
        previous_h_sum = self.__neighbor_sum(self.__x,
                                             self.__vertical_graph)
        # ``u_sum'' indicates U_{i}
        previous_u_sum = np.array([self.__sum_vector(self.__x, i) for i in range(self.__max_value+1)])

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

        print('Start the estimation')

        # Estimate parameters
        self.estimate(
            previous_v_sum, previous_h_sum, previous_u_sum,
            missing_indexes, test_data)

        # Reestimate vector x on missing indices with calculated parameters
        change_rate = 0
        ITER_REESTIMATE = 2595
        BURN_IN = 100
        SAMPLING_INTERVAL = 5
        sampled_xs = np.zeros(
            ((ITER_REESTIMATE-BURN_IN)//SAMPLING_INTERVAL+2, len(self.__x)),
            int)
        sampled_xs[0, :] = self.__x.copy()
        previous_x = self.__x.copy()

        for i in range(ITER_REESTIMATE+1):
            new_x = previous_x.copy()
            for j in missing_indexes:
                probs = np.zeros(self.__max_value + 1)

                for k in range(self.__max_value + 1):
                    t = self.__sum_vector(new_x[self.__vertical_graph.neighbors(j)], k)
                    s = self.__sum_vector(new_x[self.__horizontal_graph.neighbors(j)], k)
                    probs[k] = np.exp(
                        # self.softplus(self.__u_p)[k]
                        self.__u_p[k]
                        + self.softplus(self.__v_p)*t
                        + self.softplus(self.__h_p)*s
                    )
                denominator_p = np.sum(probs)
                probs = probs / denominator_p

                new_x[j] = np.random.choice(self.__value_list, p=probs)

            if i >= BURN_IN and i % SAMPLING_INTERVAL == 0:
                previous_idx = (i-BURN_IN) // SAMPLING_INTERVAL
                c_rate = (sampled_xs[previous_idx] != new_x).sum() / len(self.__x)
                sampled_xs[previous_idx+1, :] = new_x
                print('Change rate', c_rate)
                change_rate += c_rate

            previous_x = new_x.copy()
        print(sampled_xs)
        print('Change rate mean',
              change_rate/((ITER_REESTIMATE-BURN_IN)//SAMPLING_INTERVAL))

        # Pickup most occured feature value in the sampling on each language
        sampled_values_dists = {}
        for i in missing_indexes:
            sampled_values_dist = Counter(sampled_xs[1:, i])
            print('Sampled value distribution', i, sampled_values_dist)
            # probs = np.zeros(self.__max_value + 1)
            # for value in self.__value_list:
            #    probs[value] = sampled_values_dist[value]
            # denominator_p = np.sum(probs)
            # probs /= denominator_p
            # self.__x[i] = np.random.choice(self.__value_list, p=probs)
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

        v = self.softplus(self.__v_p)
        h = self.softplus(self.__h_p)
        u = self.__u_p # self.softplus(self.__u_p)
        estimated_parameters = OrderedDict()
        estimated_parameters['v'] = v
        estimated_parameters['h'] = h
        estimated_parameters['u'] = u.tolist()

        print('Reestimated parameters: ', v, h, u)

        info_others = OrderedDict()
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
                 previous_v_sum, previous_h_sum, previous_u_sum,
                 missing_indexes=None, test_data=None):
        """
        Estimate parameters by applying autologistic model

        Args:
            previous_v_sum, previous_h_sum, previous_u_sum
                Previous values of V_{i,j}, H_{i,j}, U_{i,j}
            missing_indexes (np vector):
                ``x``'s indexes of missing values
            test_data (np vector):
                Vector for checking the precision (hit rate)
                The precision (hit rate) is calculated
                by comparing estimated x and this (output to standard output)

        """
        sampled_x = np.zeros(len(self.__x))

        eta = 0.05 if self.__use_softplus else 0.005  # Learning rate

        likelihood_sub = 10000.0
        fudge_rate = 1.0e-8
        iteration = 0
        initial_x = self.__x.copy()

        # Corresponds to Adam's u
        beta_1 = 0.9
        beta_2 = 0.999
        m_v = 0.0
        v_v = 0.0
        m_h = 0.0
        v_h = 0.0
        m_u = 0.0
        v_u = 0.0

        iter_count = 0
        ITER_NUM = 20
        OUTER_ITER_MAX = 200
        BURN_IN_NUM = 10
        SAMPLE_NUM = ITER_NUM - BURN_IN_NUM
        cc = 1000.0

        while np.fabs(likelihood_sub) > 0.000001 and iter_count < OUTER_ITER_MAX:
            iter_count += 1

            dl_dv_p = 0
            dl_dh_p = 0
            dl_du_p = np.array([0.0 for i in range(self.__max_value+1)])

            previous_v_p = self.__v_p
            previous_h_p = self.__h_p
            previous_u_p = self.__u_p.copy()

            # Sample missing value on vector x
            if missing_indexes is None:
                dl_dv_p += previous_v_sum * SAMPLE_NUM
                dl_dh_p += previous_h_sum * SAMPLE_NUM
                dl_du_p += previous_u_sum * SAMPLE_NUM
                sampled_final = initial_x.copy()
            else:
                sampled_x = initial_x.copy()
                for j in range(ITER_NUM):
                    before_x = sampled_x.copy()

                    sampled_x = self.__sample_x(sampled_x,
                                                previous_v_p,
                                                previous_h_p,
                                                previous_u_p,
                                                missing_indexes)
                    print('CRate',
                          (before_x != sampled_x).sum() / len(missing_indexes),
                          (before_x != sampled_x).sum(), len(missing_indexes),
                          j)
                    if BURN_IN_NUM > j:
                        continue

                    ns_ph = self.__neighbor_sum(sampled_x,
                                                self.__vertical_graph,
                                                missing_indexes)
                    ns_sp = self.__neighbor_sum(sampled_x,
                                                self.__horizontal_graph,
                                                missing_indexes)
                    us = np.array([self.__sum_vector(sampled_x, i) for i in range(self.__max_value+1)])

                    dl_dv_p += ns_ph
                    dl_dh_p += ns_sp
                    dl_du_p += us

                    sampled_final = sampled_x

            # Smple full elements on vector x
            sampled_x = initial_x.copy()
            for j in range(ITER_NUM):
                before_x = sampled_x

                sampled_x = self.__sample_x(sampled_x,
                                            previous_v_p,
                                            previous_h_p,
                                            previous_u_p)
                print('CRate',
                      (before_x != sampled_x).sum() / len(self.__x),
                      (before_x != sampled_x).sum(), len(self.__x), j)

                if BURN_IN_NUM > j:
                    continue

                ns_ph = self.__neighbor_sum(sampled_x,
                                            self.__vertical_graph)
                ns_sp = self.__neighbor_sum(sampled_x,
                                            self.__horizontal_graph)
                us = np.array([self.__sum_vector(sampled_x, i) for i in range(self.__max_value+1)])

                dl_dv_p -= ns_ph
                dl_dh_p -= ns_sp
                dl_du_p -= us

            initial_x = sampled_final

            # Update parameters (Using Adam)
            likelihood_before = self.__likelihood(sampled_final)

            dl_dv_p = dl_dv_p/SAMPLE_NUM * self.d_softplus(previous_v_p)
            dl_dh_p = dl_dh_p/SAMPLE_NUM * self.d_softplus(previous_h_p)

            if self.__with_lasso:
                raise NotImplemented
                l1_u = np.array(
                    [self.__lasso(b, cc) for b in previous_u_p]) # self.softplus(previous_u_p)])
                print(dl_du_p/SAMPLE_NUM)
                dl_du_p = dl_du_p/SAMPLE_NUM - l1_u
                dl_du_p *= self.d_softplus(previous_u_p)
                print(dl_du_p)
            else:
                dl_du_p = dl_du_p/SAMPLE_NUM # * self.d_softplus(previous_u_p)
                if self.__u_lambda > 0.0:
                    print("\tdelta u {}\tL2 penalty {}".format(dl_du_p, self.__u_lambda * (previous_u_p - self.__u_p0)))
                    dl_du_p -= self.__u_lambda * (previous_u_p - self.__u_p0)

            m_v = beta_1 * m_v + (1 - beta_1) * dl_dv_p
            v_v = beta_2 * v_v + (1 - beta_2) * (dl_dv_p**2)
            m_h = beta_1 * m_h + (1 - beta_1) * dl_dh_p
            v_h = beta_2 * v_h + (1 - beta_2) * (dl_dh_p**2)
            m_u = beta_1 * m_u + (1 - beta_1) * dl_du_p
            v_u = beta_2 * v_u + (1 - beta_2) * (dl_du_p**2)

            update_targets = [
                (m_v, v_v),
                (m_h, v_h),
                (m_u, v_u)
            ]
            deltas = list()
            for m_t, v_t in update_targets:
                m_hat = m_t / (1 - beta_1 ** (iteration+1))
                v_hat = v_t / (1 - beta_2 ** (iteration+1))
                deltas.append((m_hat, v_hat))

            self.__v_p += eta * deltas[0][0] / (np.sqrt(deltas[0][1]) + fudge_rate)
            self.__h_p += eta * deltas[1][0] / (np.sqrt(deltas[1][1]) + fudge_rate)
            self.__u_p += eta * deltas[2][0] / (np.sqrt(deltas[2][1]) + fudge_rate)

            likelihood_after = self.__likelihood(sampled_final)
            likelihood_sub = likelihood_after - likelihood_before

            iteration += 1
            # p_tl = np.array([self.__h_p, self.__v_p])
            print(iteration, likelihood_sub,
                  self.softplus(self.__v_p),
                  self.softplus(self.__h_p),
                  self.__u_p) # self.softplus(self.__u_p))

            if test_data is None:
                continue
            accuracy = (sampled_final[test_data['target_indexes']] == test_data['answers']).sum() / len(test_data['answers'])
            print('Accuracy:', accuracy)

    def __lasso(self, w, c):
        if w > 0:
            return c
        else:
            print('NG')
        return 0

    def __neighbor_sum(self, vector, neighbor_graph, target_indexes=None):
        concordants = 0

        for i, a in enumerate(vector):
            for j in neighbor_graph.neighbors(i):
                if i >= j:
                    continue
                if a == vector[j]:
                    concordants += neighbor_graph[i][j]['weight']

        return concordants

    def __sum_vector(self, vector, value):
        return (vector == value).sum()

    def __sample_x(self, x_init, v_p, h_p, u_p, missing_indexes=None):
        x_ret = x_init.copy()

        if missing_indexes is None:
            # Estimate full elements
            target_indexes = list(range(len(x_ret)))
        else:
            # If given missing_indexes, estimate missing elements only
            target_indexes = missing_indexes.copy()

        np.random.shuffle(target_indexes)
        # p_tl = np.array([h, v])
        for i in target_indexes:
            probs = np.zeros(self.__max_value+1)
            for j in range(self.__max_value+1):
                t = self.__sum_vector(x_ret[self.__vertical_graph.neighbors(i)], j)
                s = self.__sum_vector(x_ret[self.__horizontal_graph.neighbors(i)], j)
                probs[j] = np.exp(
                    u_p[j] # self.softplus(u_p)[j]
                    + self.softplus(v_p)*t
                    + self.softplus(h_p)*s)

            denominator_p = np.sum(probs)
            probs = probs / denominator_p

            x_ret[i] = np.random.choice(self.__value_list, p=probs)
        return x_ret

    def __likelihood(self, x):
        t = self.__neighbor_sum(x, self.__vertical_graph)
        s = self.__neighbor_sum(x, self.__horizontal_graph)

        u = np.array([self.__sum_vector(x, i) for i in range(self.__max_value+1)])
        u_sum = sum([self.__u_p[i]*u[i] for i in range(self.__max_value+1)])
        # u_sum = sum([self.softplus(self.__u_p)[i]*u[i] for i in range(self.__max_value+1)])

        return u_sum \
            + self.softplus(self.__v_p)*t \
            + self.softplus(self.__h_p)*s

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

    def d_softplus(self, x):
        if self.__use_softplus:
            return 1. / (1 + np.exp(-x))
        return 1

    def softplus(self, x):
        if self.__use_softplus:
            return np.log(1 + np.exp(x))
        return x
