# TCG-MTS
"Temporal Gaussian Copula For Clinical Multivariate Time Series Data Imputation." In 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. 3717-3721. IEEE, 2024..
# Introduction
Multivariate time series (MTS) is particularly valuable in clinical diagnosis and disease surveillance. The imputation of the MTS is particularly applicable and challenging since the MTS typically contains irregular patterns of missing values, due to various factors such as instrument failures, interference from irrelevant data, and privacy regulations. Although existing statistical methods and deep learning methods have shown promising results in time series imputation, they failed to fully exploit the stronger temporal relationships among clinical variables when applied to the MTS. Moreover, they struggle to handle the MTS data variations in sampling density, a very common issue in MTS. To address these challenges, in this paper, we propose a Temporal Gaussian Copula Model (TGC) for three-order MTS imputation. The key idea is to leverage the Gaussian Copula to explore the cross-variable and temporal relationships based on the latent Gaussian representation. 
Subsequently, we employ an Expectation-Maximization (EM) algorithm to enhance the robustness of handling data with varying miss rates.  We finally theoretically prove our proposed TGC is equivalent to the three-order Gaussian Copula. Comprehensive experiments were conducted on three real-world MTS datasets. The results demonstrate that our TGC substantially outperforms the state-of-the-art imputation methods. Additionally, the TGC model exhibits stronger robustness to the varying missing ratios in the test dataset.
# Python Environment
gcimpute
pypots
