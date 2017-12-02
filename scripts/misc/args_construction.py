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
    parser_exp.add_argument('--distance_thres', type=int, default=1000000,
                            help='distance threshold for constructing the horizontal neighbor graph')
    parser_exp.add_argument('--distance_weighting',
                            action='store_true', default=False,
                            help='use distance weighting for the horizontal neighbor graph')
    parser_exp.add_argument('--norm_sigma', type=float, default=10.0,
                            help='standard deviation of Gaussian prior for u')
    parser_exp.add_argument('--gamma_shape', type=float, default=1.0,
                            help='shape of Gamma prior for v and h')
    parser_exp.add_argument('--gamma_scale', type=float, default=0.001,
                            help='scale of Gamma prior for v and h')
    parser_exp.add_argument('--emp-mean', dest='use_emp_mean',
                            action='store_true', default=False,
                            help='use empirical mean for Gaussian prior for u')
    parser_exp.add_argument('--use_m',
                            action='store_true', default=False,
                            help='additional factor m (both v and h)')
    parser_exp.add_argument('--init_vh', type=float, default=0.0001,
                            help="initial value of v and h")
    parser_exp.add_argument('--sample_param_weight', type=int, default=5,
                            help="# of times we perform parameter sampling per iteration")
    parser_exp.add_argument("-s", "--seed", metavar="INT", type=int, default=None,
                            help="random seed")
    parser_exp.add_argument("--cvn", metavar="INT", type=int, default=-1,
                            help="cross validation id (default: -1 (all))")
    parser_exp.set_defaults(func=execute_experiments)

    # Print features
    parser_pf.add_argument('-l', dest='language_file_path',
                           help='Path for language file (WALS csv)',
                           default=DEFAULT_LANGUAGE_PATH)
    parser_pf.set_defaults(func=print_target_features_list)

    return


def execute_experiments(args):
    print("args\t", args)
    if args.distance_weighting and args.use_m:
        print("factor m does not support distance weighting")
        exit(1)
    if args.seed is not None:
        import numpy
        print("setting seed to {}".format(args.seed))
        numpy.random.seed(args.seed)
    experiment = AutologisticExperiment(args.language_file_path,
                                        h_graph_file_path=args.hg_filename,
                                        v_graph_file_path=args.vg_filename,
                                        distance_thres=args.distance_thres,
                                        output_dir=args.output_dir)
    experiment.execute(args.min_idx, args.max_idx,
                       args.experiment_type,
                       cvn=args.cvn,
                       distance_weighting=args.distance_weighting,
                       norm_sigma=args.norm_sigma,
                       gamma_shape=args.gamma_shape,
                       gamma_scale=args.gamma_scale,
                       use_m=args.use_m,
                       use_emp_mean=args.use_emp_mean,
                       init_vh=args.init_vh,
                       sample_param_weight=args.sample_param_weight,
                       )

def print_target_features_list(args):
    experiment = AutologisticExperiment(args.language_file_path,
                                        h_graph_file_path=None,
                                        v_graph_file_path=None,
                                        output_dir='')
    experiment.print_target_features_list()
