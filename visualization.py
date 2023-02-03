import numpy as np
import matplotlib.pyplot as plt


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