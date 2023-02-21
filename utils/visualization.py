import numpy as np
import matplotlib.pyplot as plt


def plot_curves(data, legend, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(24, 18))
    add_legend = False
    for i in range(len(data)):
        if legend[i] is None:
            ax.plot(range(1, len(data[i])+1), data[i])
        else:
            add_legend = True
            ax.plot(range(1, len(data[i])+1), data[i], label=legend[i])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if add_legend:
        ax.legend()
    return fig


def scatter_plot(clean, robust):
    fig, ax = plt.subplots()
    ax.scatter(clean, robust, color='g', s=100, marker='+')
    ax.set_title('Robust accuracy wrt benign accuracy')
    ax.set_xlabel('Benign accuracy')
    ax.set_ylabel('Robust accuracy')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 50])
    ax.set_aspect('equal')
    return fig, ax


def trace_plot(eta, clean, robust):
    clean1 = clean[1:12]
    clean2 = clean[12:]
    clean_mean = (clean1 + clean2) / 2
    robust1 = robust[1:12]
    robust2 = robust[12:]
    robust_mean = (robust1 + robust2)/2
    fig, ax = plt.subplots()
    ax.scatter(eta[5], clean[0], color='g', s=100, marker='+')
    ax.scatter(eta[5], robust[0], color='g', s=100, marker='+')
    ax.scatter(eta, clean1, color='b', s=100, marker='+')
    ax.scatter(eta, clean2, color='b', s=100, marker='+')
    ax.plot(eta, clean_mean, color='b', label='Benign')
    ax.scatter(eta, robust1, color='r', s=100, marker='+')
    ax.scatter(eta, robust2, color='r', s=100, marker='+')
    ax.plot(eta, robust_mean, color='r', label='PGD')
    ax.set_title('Trace plot')
    ax.set_xlabel('Eta*1e-7 + 5e-6')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([-0.5, 10.5])
    ax.set_ylim([0, 100])
    ax.legend()
    return fig, ax


def main():
    eta = np.loadtxt('outputs/eta.txt')
    clean = np.loadtxt('outputs/clean.txt')
    robust = np.loadtxt('outputs/robust.txt')
    # _ = scatter_plot(clean, robust)
    _ = trace_plot(eta, clean, robust)
    plt.show()


if __name__ == '__main__':
    main()