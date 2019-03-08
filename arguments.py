import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset',
        dest='dataset',
        type=str.lower,
        choices=['mnist', 'fashion', 'cifar'],
        default=None,
        help='Dataset')
    parser.add_argument(
        '-t',
        '--task',
        dest='task',
        type=str.lower,
        choices=[
            'baseline', 'dimension', 'trainsize', 'visualization',
            'plot_dimension', 'plot_trainsize', 'plot_datasets', 'plot_models',
            'all'
        ],
        default='all',
        help='Task')
    parser.add_argument(
        '--start_dim',
        dest='start_dim',
        type=int,
        default='1',
        help='start dimension for "dimension" task')
    parser.add_argument(
        '--n_dim',
        dest='n_dim',
        type=int,
        default='20',
        help=
        'number of dimension (end_dim = start_dim + n_dim) for "dimension" task'
    )
    parser.add_argument(
        '--concatenate',
        dest='concat',
        action='store_true',
        help='concatenate result files into one file')
    parser.add_argument(
        '-c',
        '--classifiers',
        dest='classifiers_name',
        type=str.lower,
        choices=['logistic', 'lda', 'qda', 'knn'],
        default=['logistic', 'lda', 'qda', 'knn'],
        nargs='*',
        help='List of classifiers to use')

    parser.add_argument(
        '-m',
        '--methods',
        dest='methods_name',
        type=str.lower,
        choices=['pca', 'isomap', 'dae', 'vae'],
        default=['pca', 'isomap', 'dae', 'vae'],
        nargs='*',
        help='List of classifiers to use')

    parser.add_argument(
        '--dummy',
        dest='dummy',
        action='store_true',
        help='Use a dummy dataset (first 1000 values)')
    return parser.parse_args()