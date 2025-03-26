## Temporal Gaussian Copula For Clinical Multivariate Time Series Data Interpolation (BIBM 2024)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/MVL-Lab/TGC-MTS)
[![Visits Badge](https://badges.pufler.dev/visits/MVL-Lab/TGC-MTS)](https://badges.pufler.dev/visits/MVL-Lab/SigRL)
![GitHub forks](https://img.shields.io/github/forks/MVL-Lab/TGC-MTS?color=blue&label=Forks) 


# Introduction
Multivariate time series (MTS) is particularly valuable in clinical diagnosis and disease surveillance. The imputation of the MTS is particularly applicable and challenging since the MTS typically contains irregular patterns of missing values, due to various factors such as instrument failures, interference from irrelevant data, and privacy regulations. Although existing statistical methods and deep learning methods have shown promising results in time series imputation, they failed to fully exploit the stronger temporal relationships among clinical variables when applied to the MTS. Moreover, they struggle to handle the MTS data variations in sampling density, a very common issue in MTS. To address these challenges, in this paper, we propose a Temporal Gaussian Copula Model (TGC) for three-order MTS imputation. The key idea is to leverage the Gaussian Copula to explore the cross-variable and temporal relationships based on the latent Gaussian representation. 
Subsequently, we employ an Expectation-Maximization (EM) algorithm to enhance the robustness of handling data with varying miss rates.  We finally theoretically prove our proposed TGC is equivalent to the three-order Gaussian Copula. Comprehensive experiments were conducted on three real-world MTS datasets. The results demonstrate that our TGC substantially outperforms the state-of-the-art imputation methods. Additionally, the TGC model exhibits stronger robustness to the varying missing ratios in the test dataset.
# Python Environment
gcimpute
pypots
