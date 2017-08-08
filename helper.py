import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def clusterPlot(clusters: np.ndarray, filename: str, sep="|") -> None:
    # since clusters may start from zero or one
    first_cluster = np.min(clusters)
    last_cluster = np.max(clusters)
    output = "Cluster{0}Count{0}Genes\n".format(sep)
    for c in range(first_cluster, last_cluster + 1):
        genes_in_cluster = np.where(clusters == c)[0]
        output += "{1}{0}{2}{0}{3}".format(sep, c, genes_in_cluster.size, genes_in_cluster).replace("\n", "") + "\n"
    
    with open(filename, "w") as f:
        f.write(output)
        

def importMatlabOutput(filename: str, variable_name: str) -> np.ndarray:
    data_dict = loadmat(filename)
    return data_dict[variable_name].ravel()


def biclusterToPNG(bicluster: np.ndarray, handle):
    (width, height) = bicluster.shape
    img = np.zeros((width, height, 3))
    img[:, :, 0] = bicluster*255
    img[:, :, 1] = 255 - 255*bicluster
    handle.imshow(img, interpolation='none')
    # plt.imshow(img, interpolation='none')
    # plt.title(title)
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.ion()
    # plt.show()
