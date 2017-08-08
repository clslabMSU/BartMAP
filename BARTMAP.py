from FuzzyART import FuzzyConfig, FuzzyART
import numpy as np
import tkinter as tk


class BARTMAP:

    def __init__(self, ARTa_config: FuzzyConfig, ARTb_config: FuzzyConfig):
        self.ARTa = FuzzyART(ARTa_config)
        self.ARTb = FuzzyART(ARTb_config)

        # if defined, populates clusters to listboxes
        self.geneClusterListbox = None
        self.sampleClusterListbox = None


    def train(self, correlation_threshold: float, rho_step: float):

        # initial training of ARTa module
        weights_ARTa, clusters_ARTa, num_clusters_ARTa = self.ARTa.train(should_print=True)

        num_gene, num_samples = self.ARTa.config.data.shape
        mean_gene_cluster = np.zeros((num_samples, num_clusters_ARTa))

        # stores the number of genes in the cluster defined by the index in num_gene_cluster
        num_gene_cluster = np.zeros((num_clusters_ARTa))
        num_clusters_ARTb = 0

        for i in range(num_gene):
            num_gene_cluster[clusters_ARTa[i]] += 1
            for j in range(num_samples):
                mean_gene_cluster[j, clusters_ARTa[i]] += self.ARTa.config.data[i, j]

        for j in range(num_clusters_ARTa):
            mean_gene_cluster[:, j] /= num_gene_cluster[j]

        gene_cluster = np.zeros((num_clusters_ARTa, int(np.max(num_gene_cluster))), dtype=np.int32)
        count = np.ones((num_clusters_ARTa))

        for i in range(num_gene):
            gene_cluster[clusters_ARTa[i], int(count[clusters_ARTa[i]]) - 1] = i + 1
            count[clusters_ARTa[i]] += 1

        cluster_sample = np.zeros((num_samples), dtype=np.int32)
        sample_data_backup = self.ARTb.config.data
        sample_rho_start = self.ARTb.config.rho
        for i in range(num_samples):
            print("[STATUS] Processing sample %d" % (i + 1))
            vg = 0
            while vg == 0:
                self.ARTb.config.data = sample_data_backup[i, :].reshape((1,num_samples))
                weights_ARTb, clusters_ARTb, num_clusters_ARTb = self.ARTb.train()
                cluster_sample[i] = clusters_ARTb
                correlation = None  # will be an ndarray defined through loop iteration

                # assign the sample to an existing cluster
                if self.ARTb.config.num_clusters == num_clusters_ARTb:
                    num_cor = -1
                    for j in range(i):
                        if cluster_sample[i] == cluster_sample[j]:
                            num_cor += 1
                            if correlation is None:
                                correlation = np.empty((num_clusters_ARTa)).reshape((1,num_clusters_ARTa))
                            else:
                                correlation = np.vstack((correlation, np.empty((num_clusters_ARTa))))
                            for k in range(num_clusters_ARTa):
                                numerator, sum1, sum2, m = 0, 0, 0, 1
                                while m <= np.max(num_gene_cluster) and gene_cluster[k, m - 1] > 0 and num_gene_cluster[k] > 1:
                                    current_gene = gene_cluster[k, m - 1] - 1
                                    numerator += (sample_data_backup[current_gene, i] - mean_gene_cluster[i, k]) * \
                                                 (sample_data_backup[current_gene, j] - mean_gene_cluster[j, k])
                                    sum1 += sample_data_backup[current_gene, i] * sample_data_backup[current_gene, i]
                                    sum2 += sample_data_backup[current_gene, j] * sample_data_backup[current_gene, j]
                                    m += 1
                                correlation[num_cor, k] = numerator / (sum1**0.5 * sum2**0.5)
                    correlation_mean = np.mean(correlation, axis=0)  # verify axis is correct
                    cor_flag = 0
                    for k in range(num_clusters_ARTa):
                        if abs(correlation_mean[k]) > correlation_threshold:
                            cor_flag = 1
                            break
                    if cor_flag == 1:
                        # save k value here
                        self.ARTb.config.weights = weights_ARTb
                        self.ARTb.config.num_clusters = num_clusters_ARTb
                        vg = 1
                    else:
                        self.ARTb.config.rho += rho_step
                        if self.ARTb.config.rho > 1:
                            self.ARTb.config.rho = 1

                    del correlation
                    del correlation_mean

                else:
                    # new cluster has been created
                    self.ARTb.config.weights = weights_ARTb
                    self.ARTb.config.num_clusters = num_clusters_ARTb
                    vg = 1


        gene_cluster = [[] for i in range(num_clusters_ARTa)]
        for i in range(num_gene):
            gene_cluster[clusters_ARTa[i]].append(i)

        sample_cluster = [[] for i in range(num_clusters_ARTb)]
        for i in range(num_samples):
            sample_cluster[cluster_sample[i]].append(i)


        print("BARTMAP found {0} sample clusters with vigilance parameter {1}.".format(num_clusters_ARTb, self.ARTb.config.rho))

        print("BARTMAP found {0} gene clusters with vigilance parameter {1}.".format(num_clusters_ARTa, self.ARTa.config.rho))
        self.ARTb.config.rho = sample_rho_start

        if self.geneClusterListbox is not None:
            for g in range(len(gene_cluster)):
                self.geneClusterListbox.insert(tk.END, "Gene Cluster %d" % (g + 1))

        if self.sampleClusterListbox is not None:
            for s in range(len(sample_cluster)):
                self.sampleClusterListbox.insert(tk.END, "Sample Cluster %d" % (s + 1))

        return (gene_cluster, sample_cluster)


    def setGeneListbox(self, listbox: tk.Listbox):
        self.geneClusterListbox = listbox


    def setSampleListbox(self, listbox: tk.Listbox):
        self.sampleClusterListbox = listbox