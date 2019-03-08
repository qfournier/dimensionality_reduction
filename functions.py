''' All the functions used except the networks

Attributes:
    ISOMAP_REDUCED_SIZE (int): Maximum number of example to consider for Isomap
    args.methods_name (list):  List of the projection methods name
    len(args.classifiers_name) (TYPE):  Number of classifiers
    len(args.methods_name) (TYPE):  Number of projection methods
'''
import os
import csv
import time
import math
import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from autoencoders import DAE, VAE
from keras.datasets import mnist, fashion_mnist, cifar10
from sklearn import model_selection
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from arguments import get_args
args = get_args()

ISOMAP_REDUCED_SIZE = 10000


def create_methods(latent_dim, input_shape, batch_size, methods_name):
    '''Create a list of methods to project the data. Needs to be fitted

    Args:
        latent_dim (TYPE): Size of the projection
        input_shape (TYPE): Size of the input (for networks)
        batch_size (TYPE): Size of the batch (for networks)

    Returns:
        TYPE: List of projection methods
    '''
    methods = []
    if 'pca' in methods_name:
        methods.append(PCA(n_components=latent_dim))
    if 'isomap' in methods_name:
        methods.append(
            Isomap(n_neighbors=10, n_components=latent_dim, n_jobs=-1))
    if 'dae' in methods_name:
        methods.append(
            DAE(input_shape, batch_size=batch_size, latent_dim=latent_dim))
    if 'vae' in methods_name:
        methods.append(
            VAE(input_shape, batch_size=batch_size, latent_dim=latent_dim))
    return methods


def create_classifiers(x_train, y_train, classifiers_name):
    '''Return a list of trained classifier. The best number of neighbors for KNN
    is determined by random search.

    Args:
        x_train (TYPE): projected data to train the classifiers on
        y_train (TYPE): labels

    Returns:
        TYPE: List of trained classifiers
    '''
    classifiers = []
    if 'logistic' in classifiers_name:
        classifiers.append(
            LogisticRegression(
                solver='lbfgs',
                max_iter=9999,
                multi_class='multinomial',
                n_jobs=-1).fit(x_train, y_train))

    if 'lda' in classifiers_name:
        classifiers.append(LinearDiscriminantAnalysis().fit(x_train, y_train))

    if 'qda' in classifiers_name:
        classifiers.append(QuadraticDiscriminantAnalysis().fit(
            x_train, y_train))

    if 'knn' in classifiers_name:
        param_dist = {
            "n_neighbors":
            [i for i in range(1, int(math.sqrt(x_train.shape[0])))]
        }
        random_search = RandomizedSearchCV(
            KNeighborsClassifier(),
            param_distributions=param_dist,
            n_iter=60,
            cv=5,
            n_jobs=-1)
        classifiers.append(random_search.fit(x_train, y_train))
    return classifiers


def baseline(x_train, y_train, x_test, y_test, classifiers_name):
    file = open('./results/{}_baseline.csv'.format(args.dataset), 'w')
    writer = csv.writer(file)
    # evaluate baseline
    classifiers = create_classifiers(x_train, y_train, classifiers_name)
    # write the baseline
    for c, name in zip(classifiers, args.classifiers_name):
        writer.writerow([name, c.score(x_test, y_test)])
    file.close()


def dimension_effect(x_train,
                     y_train,
                     x_test,
                     y_test,
                     latent_dim,
                     classifiers_name,
                     methods_name,
                     iterations=10,
                     name=''):
    '''Accuracy of the classifiers in function of the size of the projection.

    Args:
        x_train (TYPE): Training examples
        y_train (TYPE): Training labels
        x_test (TYPE): Test examples
        y_test (TYPE): Test labels
        latent_dim (TYPE): List of projection dimensions
        iterations (int, optional): Number of times to train all methods and
            classifiers (in order to obtain a mean and a std of the mean)
        name (str, optional): name of the csv file to write
    '''
    print('\nAccuracy and training time w.r.t. the projection size')
    # create file
    file = open(
        './results/{}_dimension[{:02d},{:02d}].csv'.format(
            name, latent_dim[0], latent_dim[-1]), 'w')
    writer = csv.writer(file)
    # write header
    header = ['Dimension']
    for i in args.methods_name:
        header += [i + ' Time', i + ' Standard Deviation']
        for j in args.classifiers_name:
            header += [j + ' Accuracy', j + ' Standard Deviation']
    writer.writerow(header)
    file.flush()
    # for each dimensions
    for i in latent_dim:
        print('Dimension {}'.format(i))
        # list of accuracies
        row = [[] for _ in range(
            len(args.methods_name) * (len(args.classifiers_name) + 1))]
        # evaluate the accuracy 'iteration' times
        for j in range(iterations):
            print('\tIteration {} / {}'.format(j + 1, iterations))
            methods = create_methods(i, x_train.shape[1], 100, methods_name)
            for k, method in enumerate(methods):
                start = time.time()
                # if isomap, do not use all the data to train
                if isinstance(
                        method,
                        Isomap) and x_train.shape[0] > ISOMAP_REDUCED_SIZE:
                    x_train_reduced, _, y_train_reduced, _ = train_test_split(
                        x_train,
                        y_train,
                        train_size=ISOMAP_REDUCED_SIZE,
                        stratify=y_train,
                        random_state=j)
                    model = method.fit(x_train_reduced, y_train_reduced)
                else:
                    model = method.fit(x_train, y_train)
                row[k * (len(args.classifiers_name) + 1)].append(time.time() -
                                                                 start)
                classifiers = create_classifiers(
                    model.transform(x_train), y_train, classifiers_name)
                for l, classifier in enumerate(classifiers):
                    row[k * (len(args.classifiers_name) + 1) + l + 1].append(
                        classifier.score(model.transform(x_test), y_test))
        # evaluated mean and standard deviation of the mean
        new_row = [i]
        for r in row:
            new_row.append(float(np.mean(r)))
            new_row.append(float(np.std(r)))
        writer.writerow(new_row)
        file.flush()
    file.close()


def trainsize_effect(x_train,
                     y_train,
                     x_test,
                     y_test,
                     train_size,
                     latent_dim,
                     classifiers_name,
                     methods_name,
                     iterations=10,
                     name=''):
    '''Accuracy of the classifiers in function of the training set size.

    Args:
        x_train (TYPE): Training examples
        y_train (TYPE): Training labels
        x_test (TYPE): Test examples
        y_test (TYPE): Test labels
        train_size (TYPE): List of training set size to use
        latent_dim (TYPE): Projection dimension to use
        iterations (int, optional): Number of times to train all methods and
            classifiers (in order to obtain a mean and a std of the mean)
        name (str, optional): name of the csv file to write
    '''
    print('\nAccuracy and training time w.r.t. the training set size')
    header = ['Train Size']
    for i in args.methods_name:
        header += [i + ' Time', i + ' Standard Deviation']
        for j in args.classifiers_name:
            header += [j + ' Accuracy', j + ' Standard Deviation']
    for j in args.classifiers_name:
        header += [j + ' Baseline', j + ' Standard Deviation'] * len(
            args.classifiers_name)

    file = open('./results/{}_trainsize.csv'.format(name), 'w')
    writer = csv.writer(file)
    writer.writerow(header)
    for i in train_size:
        print('Taille {:.0f}'.format(i))
        row = [[] for _ in range(
            len(args.methods_name) * (len(args.classifiers_name) + 1) +
            len(args.classifiers_name))]
        for j in range(iterations):
            print('\tPasse {} / {}'.format(j + 1, iterations))
            # Create a reduced trainset (multiple of 10)
            x_train_reduced, _, y_train_reduced, _ = train_test_split(
                x_train,
                y_train,
                train_size=i,
                stratify=y_train,
                random_state=j)
            methods = create_methods(latent_dim, x_train.shape[1], 5,
                                     methods_name)
            for k, method in enumerate(methods):
                start = time.time()
                model = method.fit(x_train_reduced, y_train_reduced)
                row[k * (len(args.classifiers_name) + 1)].append(time.time() -
                                                                 start)
                classifiers = create_classifiers(
                    model.transform(x_train), y_train, classifiers_name)
                for l, classifier in enumerate(classifiers):
                    row[k * (len(args.classifiers_name) + 1) + l + 1].append(
                        classifier.score(model.transform(x_test), y_test))
            classifiers = create_classifiers(x_train, y_train,
                                             classifiers_name)
            # add baselines
            for l, c in enumerate(classifiers):
                row[-len(args.classifiers_name) + l].append(
                    c.score(x_test, y_test))
        new_row = [i]
        for r in row:
            new_row.append(float(np.mean(r)))
            new_row.append(float(np.std(r)))
        writer.writerow(new_row)
        file.flush()
    file.close()


# plot the projection in 2D
def visualization(x_train, y_train, x_test, y_test, train_size, name,
                  methods_name):
    '''Plot the projection in 2D for each methods

    Args:
        x_train (TYPE): Training examples
        y_train (TYPE): Training labels
        x_test (TYPE): Test examples
        y_test (TYPE): Test labels
        train_size (TYPE): Size of the training set
        name (TYPE): name of the figure
    '''
    x_train_reduced, _, y_train_reduced, _ = train_test_split(
        x_train, y_train, train_size=train_size, stratify=y_train)

    methods = create_methods(2, x_train.shape[1], 100, methods_name)

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    for i, method in enumerate(methods):
        model = method.fit(x_train_reduced, y_train_reduced)
        prediction = model.transform(x_test)
        plt.subplot(2, 2, i + 1)
        plt.title('PCA')
        plt.scatter(
            prediction[:, 0], prediction[:, 1], c=y_test, cmap='tab10', s=0.1)
    cmap = mpl.cm.get_cmap('tab10')
    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label=str(i),
               markerfacecolor=cmap(i),
               markersize=10) for i in range(10)
    ]
    fig.legend(handles=legend_elements, frameon=False)
    plt.savefig('figures/{}_visualization_{}.pdf'.format(name, train_size))
    plt.close()


def plot_dimesion(name, data_size, title=True):
    '''Plot effect of dimensions on accuracy and train time.

    Args:
        name (TYPE): Name of the file
    '''
    baseline = []
    if os.path.isfile('./results/{}_baseline.csv'.format(args.dataset)):
        file = open('./results/{}_baseline.csv'.format(name), 'r')
        reader = csv.reader(file)
        baseline = [
            float(row[1]) for row in reader if row[0] in args.classifiers_name
        ]
        file.close()
    file = open('./results/{}_dimension.csv'.format(name), 'r')
    reader = csv.reader(file)
    next(reader)
    # methods name
    data = list(zip(*[[float(value) for value in row] for row in reader]))
    file.close()
    # plot dimension effect on accuracy
    for j, c in enumerate(args.classifiers_name):
        plt.figure(figsize=(6, 4), tight_layout=True)
        if title:
            plt.title("{} Accuracy on {}".format(c.upper(), name.upper()))
        for i, l in enumerate(args.methods_name):
            x = np.asarray(data[0])
            x = x * 100 / data_size
            y = np.asarray(
                data[i * (len(args.classifiers_name) + 1) * 2 + 2 * j + 3])
            err = np.asarray(
                data[i * (len(args.classifiers_name) + 1) * 2 + 2 * j + 4])
            plt.plot(x, y, marker='o', markersize=5, label=l.upper())
            plt.fill_between(x, y - err, y + err, alpha=0.4)
        if baseline:
            plt.axhline(
                y=baseline[j],
                color='k',
                ls=':',
                zorder=1,
                label='Baseline (100%)')
        plt.xlabel('Dimension')
        plt.gca().xaxis.set_major_formatter(PercentFormatter(decimals=1))
        plt.ylabel('Accuracy')
        plt.legend(frameon=False)
        plt.savefig('figures/{}_dimension_accuracy_{}.pdf'.format(name, c))
        plt.close()

    # plot dimension effect on training time
    plt.figure(figsize=(6, 4), tight_layout=True)
    if title:
        plt.title("Training time on {}".format(name.upper()))
    for i, l in enumerate(args.methods_name):
        x = np.asarray(data[0])
        x = x * 100 / data_size
        y = np.asarray(data[i * (len(args.classifiers_name) + 1) * 2 + 1])
        err = np.asarray(data[i * (len(args.classifiers_name) + 1) * 2 + 2])
        plt.semilogy(x, y, marker='o', markersize=5, label=l.upper())
        plt.fill_between(x, y - err, y + err, alpha=0.4)
    plt.xlabel('Dimension')
    plt.gca().xaxis.set_major_formatter(PercentFormatter(decimals=1))
    plt.ylabel('Training Time (s)')
    plt.legend(frameon=False)
    plt.savefig('figures/{}_dimension_time.pdf'.format(name))
    plt.close()


def plot_trainsize(name, title=True):
    '''Plot effect of the training set size  on accuracy.

    Args:
        name (TYPE): Name of the file
    '''
    file = open('./results/{}_trainsize.csv'.format(name), 'r')
    reader = csv.reader(file)
    # skip header
    next(reader)
    # methods name
    data = list(zip(*[[float(value) for value in row] for row in reader]))
    # plot training size effect on accuracy
    for j, c in enumerate(args.classifiers_name):
        plt.figure(figsize=(6, 4), tight_layout=True)
        if title:
            plt.title("{} Accuracy on {}".format(c.upper(), name.upper()))
        for i, l in enumerate(args.methods_name):
            x = data[0]
            y = np.asarray(
                data[i * (len(args.classifiers_name) + 1) * 2 + 2 * j + 3])
            err = np.asarray(
                data[i * (len(args.classifiers_name) + 1) * 2 + 2 * j + 4])
            plt.plot(x, y, marker='o', markersize=5, label=l.upper())
            plt.fill_between(x, y - err, y + err, alpha=0.4)
        x = data[0]
        y = np.asarray(
            data[len(data) - len(args.classifiers_name) * 2 + 2 * j])
        err = np.asarray(
            data[len(data) - len(args.classifiers_name) * 2 + 2 * j + 1])
        plt.plot(x, y, c='gray', ls='-', label='Baseline (100%)')
        plt.fill_between(x, y - err, y + err, color='gray', alpha=0.25)
        plt.legend(frameon=False)
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.savefig('figures/{}_trainsize_accuracy_{}.pdf'.format(name, c))
        plt.close()

    # plot dimension effect on training time
    plt.figure(figsize=(6, 4), tight_layout=True)
    if title:
        plt.title("Training time on {}".format(name.upper()))
    for i, l in enumerate(args.methods_name):
        x = data[0]
        y = np.asarray(data[i * (len(args.classifiers_name) + 1) * 2 + 1])
        err = np.asarray(data[i * (len(args.classifiers_name) + 1) * 2 + 2])
        plt.semilogy(x, y, marker='o', markersize=5, label=l.upper())
        plt.fill_between(x, y - err, y + err, alpha=0.4)
    plt.xlabel('Training Size')
    plt.ylabel('Training Time (s)')
    plt.legend(frameon=False)
    plt.savefig('figures/{}_trainsize_time.pdf'.format(name))
    plt.close()


def concatenate_results(path, name):
    new_file = open("{}/{}.csv".format(path, name), 'w')
    writer = csv.writer(new_file)
    skip_header = False
    for filename in sorted(glob.glob("{}/*].csv".format(path))):
        if name in filename:
            file = open(filename, 'r')
            reader = csv.reader(file)
            if skip_header:
                next(reader)
            else:
                skip_header = True
            for row in reader:
                writer.writerow(row)
            file.close()
    new_file.close()


def plot_datasets():
    fig, axes = plt.subplots(3, 10, figsize=(15, 5), tight_layout=True)
    _, (X, Y) = mnist.load_data()
    _, X, _, Y = model_selection.train_test_split(
        X, Y, test_size=10, stratify=Y)
    Y, X = zip(*sorted(zip(Y, X)))

    for i, (x, y) in enumerate(zip(X, Y)):
        axes[0][i].set_axis_off()
        axes[0][i].set_title(y, fontweight="bold")
        axes[0][i].imshow(x, cmap='gray')

    _, (X, Y) = fashion_mnist.load_data()
    _, X, _, Y = model_selection.train_test_split(
        X, Y, test_size=10, stratify=Y)
    Y, X = zip(*sorted(zip(Y, X)))

    label = {
        0: "T-shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    for i, (x, y) in enumerate(zip(X, label)):
        axes[1][i].set_axis_off()
        axes[1][i].set_title(label[y], fontweight="bold")
        axes[1][i].imshow(x, cmap='gray')

    _, (X, Y) = cifar10.load_data()
    _, X, _, Y = model_selection.train_test_split(
        X, Y, test_size=10, stratify=Y)
    Y, X = zip(*sorted(zip(Y, X)))

    label = {
        0: "Airplane",
        1: "Automobile",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck"
    }

    for i, (x, y) in enumerate(zip(X, Y)):
        axes[2][i].set_axis_off()
        axes[2][i].set_title(label[y[0]], fontweight="bold")
        axes[2][i].imshow(x)

    plt.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig("./figures/datasets.pdf")
