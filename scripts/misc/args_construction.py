from autologistic.experiment import Experiment as AutologisticExperiment

DEFAULT_LANGUAGE_PATH = 'data/wals/language.csv'

# Commands for autologistic models
def parser_for_autologistic(parent_sub_parsers):
    experiment_type_description = """\
Experiment type
- mvi   Evaluate the performance of the autologistic model
        to estimate missing value
        (calculate the accuracies of 10-fold closs validation)
- param Estimate v and h
    """
    parser = parent_sub_parsers.add_parser('autologistic')
    sub_parsers = parser.add_subparsers(dest='autologistic_commands')

    parser_exp = sub_parsers.add_parser('exp', help='Execute experiments')
    parser_exp_cpt = sub_parsers.add_parser('exp_cpt', help='Execute experiments (compatible)')
    parser_exp_ibp = sub_parsers.add_parser('exp_ibp', help='Execute experiments IBP')
    parser_pf = sub_parsers.add_parser('print-features',
                                       help='Print features to process'\
                                       'with their indices')

    # Execute experiments
    parser_exp.add_argument('min_idx', type=int,
                            help='Minimum index of the target features')
    parser_exp.add_argument('max_idx', type=int,
                            help='Maximum index of the target features')
    parser_exp.add_argument('experiment_type', type=str,
                            choices=['param', 'mvi'],
                            help=experiment_type_description)
    parser_exp.add_argument('-l', dest='language_file_path',
                            help='Path for language file (WALS csv)',
                            default=DEFAULT_LANGUAGE_PATH)
    parser_exp.add_argument('--hg', dest='hg_filename',
                            help='Path for horizontal graph file',
                            default=None)
    parser_exp.add_argument('--vg', dest='vg_filename',
                            help='Path for vertical graph file',
                            default=None)
    parser_exp.add_argument('-o', dest='output_dir',
                            help='Directory for output files', default='')
    parser_exp.add_argument('--with-lasso', dest='with_lasso',
                            action='store_true',
                            help='With LASSO', default=False)
    parser_exp.add_argument('--use-softplus', dest='use_softplus',
                            action='store_true',
                            help='Apply softplus', default=False)
    parser_exp.add_argument('--u_lambda', type=float, default=0.0,
                            help='L2 regularization for u (default: 0.0)')
    parser_exp.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                            help="random seed")
    parser_exp.set_defaults(func=execute_experiments)

    # Print features
    parser_pf.add_argument('-l', dest='language_file_path',
                           help='Path for language file (WALS csv)',
                           default=DEFAULT_LANGUAGE_PATH)
    parser_pf.set_defaults(func=print_target_features_list)

    return


def execute_experiments(args):
    if args.seed is not None:
        import numpy
        print("setting seed to {}".format(args.seed))
        numpy.random.seed(args.seed)
    experiment = AutologisticExperiment(args.language_file_path,
                                        h_graph_file_path=args.hg_filename,
                                        v_graph_file_path=args.vg_filename,
                                        output_dir=args.output_dir)
    experiment.execute(args.min_idx, args.max_idx,
                       args.experiment_type, with_lasso=args.with_lasso,
                       use_softplus=args.use_softplus, u_lambda=args.u_lambda)


def execute_experiments_cpt(args):
    if args.seed is not None:
        import numpy
        print("setting seed to {}".format(args.seed))
        numpy.random.seed(args.seed)
    experiment = AutologisticExperimentCpt(args.language_file_path,
                                           args.feature_file_path,
                                           h_graph_file_path=args.hg_filename,
                                           v_graph_file_path=args.vg_filename,
                                           output_dir=args.output_dir)
    experiment.execute(args.min_idx, args.max_idx,
                       args.experiment_type, cvi=args.cvi, with_lasso=args.with_lasso,
                       use_softplus=args.use_softplus, u_lambda=args.u_lambda)


def execute_experiments_ibp(args):
    if args.seed is not None:
        import numpy
        print("setting seed to {}".format(args.seed))
        numpy.random.seed(args.seed)
    experiment = AutologisticExperimentIBP(args.language_file_path,
                                           h_graph_file_path=args.hg_filename,
                                           v_graph_file_path=args.vg_filename,
                                           output_dir=args.output_dir)
    experiment.execute(args.min_idx, args.max_idx,
                       args.experiment_type,
                       with_lasso=args.with_lasso,
                       use_softplus=args.use_softplus, u_lambda=args.u_lambda)


def print_target_features_list(args):
    experiment = AutologisticExperiment(args.language_file_path,
                                        h_graph_file_path=None,
                                        v_graph_file_path=None,
                                        output_dir='')
    experiment.print_target_features_list()
