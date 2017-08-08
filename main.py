from BARTMAP import BARTMAP
from FuzzyART import FuzzyConfig
from helper import biclusterToPNG
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from multiprocessing.pool import ThreadPool
import numpy as np
from scipy.io import loadmat
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.scrolledtext as tkst


class App:
    def __init__(self, master: tk.Tk):

        # initialize some BARTMAP properties
        self.input_data = None
        self.gene_clusters = None
        self.sample_clusters = None
        self.ARTa_config = None
        self.ARTb_config = None

        self.thread_pool = ThreadPool(processes=1)
        self.bartmap_thread = None

        # Root window configuration
        self.master = master
        master.title("PyClustering BARTMAP")
        master.geometry("1000x600+100+100")

        # Horizontal divider
        self.panedWindowVertical = tk.PanedWindow(orient=tk.VERTICAL)
        self.panedWindowVertical.pack(fill=tk.BOTH, expand=1)

        # Vertical divider
        self.panedWindowHorizontal = tk.PanedWindow(orient=tk.HORIZONTAL)
        self.panedWindowVertical.add(self.panedWindowHorizontal)

        # Left frame for configuration options
        self.configFrame = tk.Frame(self.panedWindowHorizontal)
        self.configFrame.columnconfigure(1, weight=1)
        self.panedWindowHorizontal.add(self.configFrame)

        # Middle-left frame for gene cluster results
        self.geneResultsListBox = tk.Listbox(self.panedWindowHorizontal)
        self.geneResultsListBox.bind("<Double-Button-1>", self.onClickGeneResults)
        self.panedWindowHorizontal.add(self.geneResultsListBox)

        # Middle-right frame for sample cluster results
        self.sampleResultsListBox = tk.Listbox(self.panedWindowHorizontal)
        self.sampleResultsListBox.bind("<Double-Button-1>", self.onClickSampleResults)
        self.panedWindowHorizontal.add(self.sampleResultsListBox)

        # Right frame for image preview
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.gca().axes.get_xaxis().set_visible(False)
        self.figure.gca().axes.get_yaxis().set_visible(False)
        self.implot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.panedWindowHorizontal)
        self.canvas.show()
        self.panedWindowHorizontal.add(self.canvas.get_tk_widget())

        # Status bar at bottom of screen
        self.statusBar = tkst.ScrolledText(master, state=tk.DISABLED)
        self.panedWindowVertical.add(self.statusBar)
        sys.stdout = StdoutRedirector(sys.stdout, self.statusBar)

        # Gene settings
        self.labelGeneSettings = tk.Label(self.configFrame, text="Gene Settings", font=("Courier", 12))
        self.labelGeneSettings.grid(row=0, column=0, columnspan=2)

        self.geneAlpha = tk.DoubleVar(value=0.001)
        self.labelGeneAlpha = tk.Label(self.configFrame, text="Alpha: ")
        self.inputGeneAlpha = tk.Entry(self.configFrame, textvariable=self.geneAlpha)
        self.labelGeneAlpha.grid(row=1, column=0)
        self.inputGeneAlpha.grid(row=1, column=1, sticky=tk.W+tk.E)

        self.geneBeta = tk.DoubleVar(value=1.0)
        self.labelGeneBeta = tk.Label(self.configFrame, text="Beta: ")
        self.inputGeneBeta = tk.Entry(self.configFrame, textvariable=self.geneBeta)
        self.labelGeneBeta.grid(row=2, column=0)
        self.inputGeneBeta.grid(row=2, column=1, sticky=tk.W+tk.E)

        self.geneRho = tk.DoubleVar(value=0.15)
        self.labelGeneRho = tk.Label(self.configFrame, text="Rho: ")
        self.inputGeneRho = tk.Entry(self.configFrame, textvariable=self.geneRho)
        self.labelGeneRho.grid(row=3, column=0)
        self.inputGeneRho.grid(row=3, column=1, sticky=tk.W+tk.E)

        self.geneMaxEpochs = tk.IntVar(value=1000)
        self.labelGeneMaxEpochs = tk.Label(self.configFrame, text="Epochs: ")
        self.inputGeneMaxEpochs = tk.Entry(self.configFrame, textvariable=self.geneMaxEpochs)
        self.labelGeneMaxEpochs.grid(row=4, column=0)
        self.inputGeneMaxEpochs.grid(row=4, column=1, sticky=tk.W+tk.E)

        # Sample settings
        self.labelGeneSettings = tk.Label(self.configFrame, text="Sample Settings", font=("Courier", 12))
        self.labelGeneSettings.grid(row=5, column=0, columnspan=2)

        self.sampleAlpha = tk.DoubleVar(value=0.001)
        self.labelGeneAlpha = tk.Label(self.configFrame, text="Alpha: ")
        self.inputGeneAlpha = tk.Entry(self.configFrame, textvariable=self.sampleAlpha)
        self.labelGeneAlpha.grid(row=6, column=0)
        self.inputGeneAlpha.grid(row=6, column=1, sticky=tk.W+tk.E)

        self.sampleBeta = tk.DoubleVar(value=1.0)
        self.labelGeneBeta = tk.Label(self.configFrame, text="Beta: ")
        self.inputGeneBeta = tk.Entry(self.configFrame, textvariable=self.sampleBeta)
        self.labelGeneBeta.grid(row=7, column=0)
        self.inputGeneBeta.grid(row=7, column=1, sticky=tk.W+tk.E)

        self.sampleRho = tk.DoubleVar(value=0.15)
        self.labelGeneRho = tk.Label(self.configFrame, text="Rho: ")
        self.inputGeneRho = tk.Entry(self.configFrame, textvariable=self.sampleRho)
        self.labelGeneRho.grid(row=8, column=0)
        self.inputGeneRho.grid(row=8, column=1, sticky=tk.W+tk.E)

        self.sampleMaxEpochs = tk.IntVar(value=1000)
        self.labelSampleMaxEpochs = tk.Label(self.configFrame, text="Epochs: ")
        self.inputSampleMaxEpochs = tk.Entry(self.configFrame, textvariable=self.sampleMaxEpochs)
        self.labelSampleMaxEpochs.grid(row=9, column=0)
        self.inputSampleMaxEpochs.grid(row=9, column=1, sticky=tk.W+tk.E)

        # BARTMAP settings
        self.labelGeneSettings = tk.Label(self.configFrame, text="BARTMAP Settings", font=("Courier", 12))
        self.labelGeneSettings.grid(row=10, column=0, columnspan=2)

        self.correlationThreshold = tk.DoubleVar(value=0.2)
        self.labelCorrelationThreshold = tk.Label(self.configFrame, text="Correlation: ")
        self.inputCorrelationThreshold = tk.Entry(self.configFrame, textvariable=self.correlationThreshold)
        self.labelCorrelationThreshold.grid(row=11, column=0)
        self.inputCorrelationThreshold.grid(row=11, column=1, sticky=tk.W+tk.E)

        self.rho_step = tk.DoubleVar(value=0.05)
        self.labelrho_step = tk.Label(self.configFrame, text="Rho Step: ")
        self.inputrho_step = tk.Entry(self.configFrame, textvariable=self.rho_step)
        self.labelrho_step.grid(row=12, column=0)
        self.inputrho_step.grid(row=12, column=1, sticky=tk.W+tk.E)

        self.labelDatafile = tk.Label(self.configFrame, text="Data: ")
        self.labelLoadedDatafile = tk.Label(self.configFrame, text="Not Selected", fg="red")
        self.labelDatafile.grid(row=13, column=0)
        self.labelLoadedDatafile.grid(row=13, column=1, sticky=tk.W)

        self.buttonLoadMATFile = tk.Button(self.configFrame, text="Select MAT File", command=self.load_mat_file)
        self.buttonLoadMATFile.grid(row=14, column=0, columnspan=2)

        self.buttonStartBARTMAP = tk.Button(self.configFrame, text="Run BARTMAP", command=self.start_BARTMAP)
        self.buttonStartBARTMAP.grid(row=15, column=0, columnspan=2)


    def load_mat_file(self):
        filename = askopenfilename(filetypes=[("MAT files", "*.mat")])
        print("[STATUS] Loading %s." % filename)
        try:
            self.input_data = loadmat(filename, matlab_compatible=True)["input_gene"]
            self.labelLoadedDatafile.config(text=filename.split("/")[-1], fg="black")
            print("[STATUS] Finished loading file.")
        except FileNotFoundError:
            print("[ERROR] The provided .mat file could not be located.")
        except KeyError:
            print("[ERROR] The variable 'input_gene' could not be located in the provided .mat file.")


    def start_BARTMAP(self):
        self.generate_ART_configs()
        correlation_threshold = self.correlationThreshold.get()
        rho_step = self.rho_step.get()
        if None not in [self.ARTa_config, self.ARTb_config, correlation_threshold, rho_step]:
            bartmap = BARTMAP(self.ARTa_config, self.ARTb_config)
            bartmap.setGeneListbox(self.geneResultsListBox)
            bartmap.setSampleListbox(self.sampleResultsListBox)
            self.geneResultsListBox.delete(0, tk.END)
            self.sampleResultsListBox.delete(0, tk.END)
            self.bartmap_thread = self.thread_pool.apply_async(bartmap.train, (correlation_threshold, rho_step))
        else:
            print("[ERROR] Configuration is not valid.")


    def generate_ART_configs(self):
        if self.input_data is not None:
            self.ARTa_config = FuzzyConfig({
                "alpha": self.geneAlpha.get(),
                "beta": self.geneBeta.get(),
                "rho": self.geneRho.get(),
                "epochs": self.geneMaxEpochs.get(),
                "prune": False,
                "verbose": False,
                "data": self.input_data
            })
            self.ARTb_config = FuzzyConfig({
                "alpha": self.sampleAlpha.get(),
                "beta": self.sampleBeta.get(),
                "rho": self.sampleRho.get(),
                "epochs": self.sampleMaxEpochs.get(),
                "prune": False,
                "verbose": False,
                "data": self.input_data
            })
        else:
            print("[ERROR] Input data has not been loaded.")


    def onClickGeneResults(self, event):
        widget = event.widget
        selection = widget.curselection()
        if len(selection) > 0 and self.bartmap_thread is not None:
            value = widget.get(selection[0])
            (self.gene_clusters, self.sample_clusters) = self.bartmap_thread.get()
            gene_cluster_index = int(value.split(" ")[-1]) - 1
            gene_cluster_values = self.input_data[np.ix_(self.gene_clusters[gene_cluster_index])]
            biclusterToPNG(gene_cluster_values, self.implot)
            self.figure.suptitle(value)
            self.canvas.draw()


    def onClickSampleResults(self, event):
        widget = event.widget
        selection = widget.curselection()
        if len(selection) > 0 and self.bartmap_thread is not None:
            value = widget.get(selection[0])
            (self.gene_clusters, self.sample_clusters) = self.bartmap_thread.get()
            sample_cluster_index = int(value.split(" ")[-1]) - 1
            sample_cluster_values = self.input_data[np.ix_(self.sample_clusters[sample_cluster_index])]
            biclusterToPNG(sample_cluster_values, self.implot)
            self.figure.suptitle(value)
            self.canvas.draw()


class IORedirector(object):
    def __init__(self, default_stdout, text_area):
        self.text_area = text_area
        self.default_stdout = default_stdout


class StdoutRedirector(IORedirector):
    def write(self, s):
        self.default_stdout.write(s)
        try:
            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, s)
            self.text_area.see(tk.END)
            self.text_area.config(state=tk.DISABLED)
        except tk.TclError:
            pass
    def flush(self):
        self.default_stdout.flush()


root = tk.Tk()
app = App(root)
root.mainloop()