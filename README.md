# ESM-Scan
Complete saturation mutagenesis of a protein using ESM

Analyze your own sequence using Google colab: https://colab.research.google.com/github/xuebingwu/ESM-Scan/blob/main/esm-scan-colab.ipynb

**Input**
* A single protein sequence

**Output**
* Data: CSV files containing the effect of each mutations. Negative means more deleterious. 
* Visualizaitons: A heatmap color coding the effect of all possible mutations (20 columns) at each amino acid in the protein (row). A box-plot along each position is also included.

<img src="https://github.com/xuebingwu/ESM-Scan/blob/main/example-output.png" height="200" align="center" style="height:240px">

**Methods**
* Please see the following preprint for more details: 
[Language models enable zero-shot prediction of the effects of mutations on protein function](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2).

**Limitations**
* A gmail account is required to run Google Colab notebooks.
* This notebook was designed for analyzing a single sequence. 
* Only sequences of length ~400aa have been tested. Longer sequences may fail due to a lack of memory.
* The first run is slow due to the need to download pre-trained ESM models.  
* GPU may not be available and running the prediction on CPU will be significantly slower. 
* Your browser can block the pop-up for downloading the result file. You can choose the `save_to_google_drive` option to upload to Google Drive instead or manually download the result file: Click on the little folder icon to the left, navigate to file: `res.zip`, right-click and select \"Download\".


**Bugs**
- If you encounter any bugs, please report the issue by emailing Xuebing Wu (xw2629 at cumc dot columbia dot edu)

**License**

* The source code of this notebook is licensed under [MIT](https://raw.githubusercontent.com/sokrypton/ColabFold/main/LICENSE).

**Acknowledgments**
- We thank the [ESM](https://github.com/facebookresearch/esm) team for developing an excellent model and open sourcing the software. 

- This notebook is modeld after the [ColabFold notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb).

