import os
import numpy as np

from functions import concatenate_results
from functions import dimension_effect
from functions import trainsize_effect
from functions import plot_datasets
from functions import visualization
from functions import plot_dimesion
from functions import plot_trainsize
from functions import baseline
from functions import load_uji
from arguments import get_args

from autoencoders import DAE, VAE

from sklearn.preprocessing import StandardScaler

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import plot_model

# SUPPRESS WARNING, SHOULD BE COMMENTED
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # Create folders
    if 'results' not in os.listdir():
        os.mkdir('./results')
    if 'figures' not in os.listdir():
        os.mkdir('./figures')

    args = get_args()

    if args.dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    elif args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    elif args.dataset == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif args.dataset == 'uji':
        x_train, y_train = load_uji("data/train.csv")
        x_test, y_test = load_uji("data/test.csv")

    if args.dataset == None:
        print('WARNING:no dataset selected')
    elif args.dataset == 'uji':
        # Standardize values (0 mean and unit variance)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        # Normalize values between 0 and 1
        x_train = (x_train.astype('float32')) / 255.
        x_test = (x_test.astype('float32')) / 255.
        # Put images into vectors
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        # Reshape label as vector
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    if args.dummy:
        x_train, y_train = x_train[:1000], y_train[:1000]

    if args.task == 'baseline' or args.task == 'all':
        baseline(
            x_train,
            y_train,
            x_test,
            y_test,
            classifiers_name=args.classifiers_name)

    if args.task == 'dimension' or args.task == 'all':
        dimension = [
            i for i in range(args.start_dim, args.start_dim + args.n_dim)
        ]
        dimension_effect(
            x_train,
            y_train,
            x_test,
            y_test,
            dimension,
            iterations=5,
            classifiers_name=args.classifiers_name,
            methods_name=args.methods_name,
            name=args.dataset)

    if args.task == 'trainsize' or args.task == 'all':
        train_size = [i * 50 for i in range(1, 21)]
        trainsize_effect(
            x_train,
            y_train,
            x_test,
            y_test,
            train_size,
            latent_dim=2,
            iterations=5,
            classifiers_name=args.classifiers_name,
            methods_name=args.methods_name,
            name=args.dataset)

    if args.task == 'visualization' or args.task == 'all':
        # Visualization of the projection in 2D (in function of the train size)
        visualization(x_train, y_train, x_test, y_test, 300, parser.dataset,
                      methods_name)

    if args.task == 'plot_dimension' or args.task == 'all':
        if args.concat:
            concatenate_results("./results",
                                "{}_dimension".format(args.dataset))
        plot_dimesion(args.dataset, x_test.shape[1], title=False)

    if args.task == 'plot_trainsize' or args.task == 'all':
        if args.concat:
            concatenate_results("results", "{}_trainsize".format(args.dataset))
        plot_trainsize(args.dataset)

    if args.task == 'plot_datasets' or args.task == 'all':
        plot_datasets()

    if args.task == 'plot_models' or args.task == 'all':
        model = DAE(28 * 28, batch_size=100, latent_dim=2)
        plot_model(
            model.autoencoder,
            show_shapes=True,
            to_file='figures/DAE_MNIST.pdf')

        model = VAE(28 * 28, batch_size=100, latent_dim=2)
        plot_model(
            model.autoencoder,
            show_shapes=True,
            to_file='figures/VAE_MNIST.pdf')