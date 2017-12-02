from sklearn.model_selection import KFold
import os
import pickle
import json
from collections import defaultdict
from collections import OrderedDict

from .horizontal_graph_generator import HorizontalGraphGenerator
from .vertical_graph_generator import VerticalGraphGenerator
from .autologistic import Autologistic
from misc.wals_csv_loader import WalsCsvLoader
import numpy as np


class Experiment(object):
    """Execute experiment(s) for the evaluation of Autologistic model

    Args:
        language_file_path (str): File path for the WALS language csv file
        v_graph_file_path (str): File path for the vertical graph
        h_graph_file_path (str): File path for the horizontal graph
        output_dir (str): Directory path for output

    Attributes:
        languages (array):
        vertical_graph ():
        horizontal_graph ():
        output_dir (str): Directory path for output

    """
    def __init__(self, language_file_path, **kwargs):
        self.fill_rate_thres = 0.1

        h_graph_file_path = kwargs['h_graph_file_path']
        v_graph_file_path = kwargs['v_graph_file_path']

        output_dir = kwargs['output_dir']
        if not output_dir:
            output_dir = os.path.join('results', 'autologistic')
        self.output_dir = output_dir

        self.languages = []
        self.__feature_values = defaultdict(set)
        self.__feature_value_map = {}
        self.__int_feature_map = {}

        self.vertical_graph = None
        self.horizontal_graph = None

        self.__init_data(
            language_file_path,
            v_graph_file_path,
            h_graph_file_path,
            distance_thres=kwargs['distance_thres'],
        )

    def __init_data(self, language_file_path,
                    v_graph_file_path='vertical_graph.pkl',
                    h_graph_file_path='horizontal_graph.pkl',
                    ph_level='genus',
                    distance_thres=1000000,
                    print_only=False):
        self.__load_languages(language_file_path, ph_level=ph_level)
        self.__create_feature_value_map()

        # Create graphs
        if v_graph_file_path and os.path.isfile(v_graph_file_path):
            with open(v_graph_file_path, 'rb') as f:
                self.vertical_graph = pickle.load(f)
        else:
            v_graph_generator = VerticalGraphGenerator()
            self.vertical_graph = v_graph_generator.generate_graph(
                self.languages, v_graph_file_path
            )

        if h_graph_file_path and os.path.isfile(h_graph_file_path):
            with open(h_graph_file_path, 'rb') as f:
                self.horizontal_graph = pickle.load(f)
        else:
            h_graph_generator = HorizontalGraphGenerator()
            self.horizontal_graph = h_graph_generator.generate_graph(
                self.languages, h_graph_file_path,
                distance_thres=distance_thres
            )

    def __load_languages(self, language_file_path, ph_level='genus'):
        loader = WalsCsvLoader()
        loaded_data = loader.load(language_file_path)
        for language in loaded_data["languages"]:
            language['ph_group'] = language[ph_level]
        self.languages = loaded_data['languages']
        self.__feature_values = loaded_data['feature_values']

    def __create_feature_value_map(self):
        """Create integer to feature raw value map
        """
        for feature_name, values in self.__feature_values.items():
            value_map = {}
            int_map = {}
            values = sorted(values)
            for i, value in enumerate(values):
                value_map[value] = i
                int_map[i] = value

            self.__feature_value_map[feature_name] = value_map
            self.__int_feature_map[feature_name] = int_map

    def print_target_features_list(self):
        """Print target features list
        """
        selected_features, xs = self.__select_target_features()
        print("Index\tFeature name")
        print('----------------------------------------------')
        for i, feature in enumerate(selected_features):
            print(str(i)+"\t"+feature)

    def execute(self, feature_idx_min, feature_idx_max,
                experiment_type, log=False,
                cvn=-1,
                distance_weighting=False,
                norm_sigma=10.0,
                gamma_shape=1.0,
                gamma_scale=0.001,
                use_m=False,
                use_emp_mean=False,
                init_vh=0.0001,
                sample_param_weight=5,
    ):
        """Execute multiclass Autologistic model

        Args:
            feature_idx_min (int): Minimum index of the target features
            feature_idx_max (int): Maximum index of the target features
            experiment_type (str):
                If ``mvi``, it evaluate the performance of
                 missing value imputation by calculate 10-fold accuracies.
                If ``param``, it estimates parameters, :math:`v`
                and :math:`h` and missing values
                by using all observed feature values
            log (bool):
                If ``True``, it outputs a log file of the experiment process
        """
        selected_features, xs = self.__select_target_features()

        for i, feature in enumerate(selected_features):
            if feature_idx_min > i:
                continue
            if feature_idx_max < i:
                break

            print('Feature:', feature, '(Index:', i, ')')
            max_feature_value = max(self.__feature_value_map[feature].values())

            if experiment_type == 'mvi':
                self.__hide_existing_values(
                    feature, xs[i], max_feature_value,
                    cvn=cvn,
                    distance_weighting=distance_weighting,
                    norm_sigma=norm_sigma,
                    gamma_shape=gamma_shape,
                    gamma_scale=gamma_scale,
                    use_m=use_m,
                    use_emp_mean=use_emp_mean,
                    init_vh=init_vh,
                    sample_param_weight=sample_param_weight,
                )
            elif experiment_type == 'param':
                model = Autologistic(
                    xs[i],
                    self.vertical_graph,
                    self.horizontal_graph,
                    max_feature_value,
                    distance_weighting=distance_weighting,
                    norm_sigma=norm_sigma,
                    gamma_shape=gamma_shape,
                    gamma_scale=gamma_scale,
                    use_m=use_m,
                    use_emp_mean=use_emp_mean,
                    init_vh=init_vh,
                    sample_param_weight=sample_param_weight,
                )
                result = model.estimate_with_missing_value()
                output_dir_param = os.path.join(
                    self.output_dir, 'param'
                )
                if not os.path.exists(output_dir_param):
                    os.makedirs(output_dir_param, exist_ok=True)
                self.__output_result(
                    feature, xs[i], result,
                    output_dir_param, hidden_indexes=[]
                )
            else:
                raise ValueError('Experiment type must be param or mvi')

    def __select_target_features(self):
        selected_features = []
        features_list = sorted(list(self.__feature_value_map.keys()))
        xs = []

        black_list = [
            "144D The Position of Negative Morphemes in SVO Languages",
            "144H NegSVO Order",
            "144H NegSVO Order",
            "144I SNegVO Order",
            "144J SVNegO Order",
            "144K SVONeg Order",
            "144P NegSOV Order",
            "144Q SNegOV Order",
            "144R SONegV Order",
            "144S SOVNeg Order",
            '144L The Position of Negative Morphemes in SOV Languages'
        ]

        for feature in features_list:
            if feature in black_list:
                continue
            # Create vector x
            estimate_targets = []
            x_ = []
            for idx, language in enumerate(self.languages):
                if language['features'][feature] == 'NA':
                    x_.append(-1)
                    estimate_targets.append(idx)
                else:
                    int_feature_value = self.__feature_value_map[feature][language['features'][feature]]
                    x_.append(int_feature_value)
            x = np.array(x_)
            fill_rate = len(np.where(x != -1)[0]) / len(x)

            if fill_rate >= self.fill_rate_thres:
                selected_features.append(feature)
                xs.append(x)

        return (selected_features, xs)

    def __hide_existing_values(self, feature, x, max_value, cvn=-1,
                               distance_weighting=False,
                               norm_sigma=10.0,
                               gamma_shape=1.0,
                               gamma_scale=0.001,
                               use_m=False,
                               use_emp_mean=False,
                               init_vh=0.0001,
                               sample_param_weight=5,
    ):
        """Execute a 10-fold cross validation experiment to
           evaluate model performance of missing value imputation
        """
        fold_num = 10
        x_original = x.copy()
        not_missing_indexes = np.where(x_original != -1)[0]
        np.random.shuffle(not_missing_indexes)

        split_unit = len(not_missing_indexes)//fold_num
        print('Split_unit', split_unit)

        k_fold = KFold(n_splits=fold_num, shuffle=True, random_state=np.random.RandomState(10))
        for cv_count, folds in enumerate(k_fold.split(not_missing_indexes)):
            if cvn >= 0 and cv_count != cvn:
                continue
            x = x_original.copy()
            _, test = folds
            test.sort()
            hidden_indexes = [not_missing_indexes[y] for y in test]

            test_data = {}
            test_data['target_indexes'] = hidden_indexes
            test_data['answers'] = x[hidden_indexes].copy()
            x[hidden_indexes] = -1

            output_dir_mvi = os.path.join(
                self.output_dir,
                'mvi', "{0:03d}".format(cv_count)
            )
            if not os.path.exists(output_dir_mvi):
                os.makedirs(output_dir_mvi, exist_ok=True)
            model = Autologistic(
                x,
                self.vertical_graph,
                self.horizontal_graph,
                max_value,
                distance_weighting=distance_weighting,
                norm_sigma=norm_sigma,
                gamma_shape=gamma_shape,
                gamma_scale=gamma_scale,
                use_m=use_m,
                use_emp_mean=use_emp_mean,
                init_vh=init_vh,
                sample_param_weight=sample_param_weight,
            )
            result = model.estimate_with_missing_value(test_data=test_data)

            self.__output_result(feature, x_original, result,
                                 output_dir_mvi, hidden_indexes)

    def __output_result(self, feature, x_original, result,
                        output_dir, hidden_indexes=[]):
        """Output experiment results as a JSON format
        """
        accuracy_exp = False
        output_json = OrderedDict()

        estimated_x, estimated_params, info_others \
            = result

        if 'accuracies' in info_others:
            accuracy_exp = True

        estimate_targets = np.where(x_original == -1)[0]

        # Generate output json
        output_json['feature'] = feature
        output_json['parameters'] = estimated_params
        if accuracy_exp:
            output_json['accuracy'] = info_others['accuracies']
        if 'sampled_params' in info_others:
            output_json['sampled_params'] = info_others['sampled_params']

        output_json['estimate_results'] = []
        for language_idx, int_value in enumerate(estimated_x):
            hidden_estimated = language_idx in hidden_indexes
            estimated = language_idx in estimate_targets

            estimate_result = OrderedDict()
            wals_language = self.languages[language_idx]
            estimate_result['name'] = wals_language['name']
            estimate_result['glottocode'] = wals_language['glottocode']
            estimate_result['iso_code'] = wals_language['iso_code']
            if accuracy_exp:
                estimate_result['is_hidden_estimated'] = hidden_estimated

            # Make distribution rates of sampled values
            if estimated or hidden_estimated:
                sampled_distribution\
                    = info_others['sampled_values_distributions'][language_idx]
                json_sampled_distribution = OrderedDict()
                sum_sampled_num = sum(sampled_distribution.values())
                for int_feature_value in sorted(sampled_distribution):
                    str_feature_value\
                        = self.__int_feature_map[feature][int_feature_value]
                    dist_rate\
                        = sampled_distribution[int_feature_value]/sum_sampled_num
                    json_sampled_distribution[str_feature_value] = dist_rate
                estimate_result['sampled_values_distributions'] \
                    = json_sampled_distribution

            if hidden_estimated:
                estimate_result['value'] \
                    = self.__int_feature_map[feature][x_original[language_idx]]
                estimate_result['original'] \
                    = self.__int_feature_map[feature][int_value]
            else:
                estimate_result['value'] \
                    = self.__int_feature_map[feature][int_value]
                estimate_result['is_original'] = not estimated
            output_json['estimate_results'].append(estimate_result)

        output_file_name = '_'.join(feature.split(' ')) + '.json'
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, 'w') as f:
            json.dump(
                output_json,
                f,
                ensure_ascii=False,
                indent=4, separators=(',', ': ')
            )
