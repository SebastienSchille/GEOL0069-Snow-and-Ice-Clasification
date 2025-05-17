<p align="center">
  <img src="/Images/ndsi_deployment_region.png" width="600" height="auto"/>
</p>

# Monitoring Glacier Retreat in the Swiss Alps Using Supervised and Unsupervised Machine Learning.

# 1.0 Project Background

Glaciers are extremely sensitive to climate change and can be studied to highlight key indicators of climate change. There has been an alarming increase in glacier retreat around the globe. This reduction can have catastrophic consequences on mountain ecosystems, freshwater supplies, tourism, and hydropower. Accurate monitoring is crucial to not only assessing the impacts of climate change but also informing adaptation strategies.

Traditional data collection methods, such as manual sample collection and interpreting signal data, can be highly time-consuming and often have limited data. Earth observation satellites launched within the Copernicus program have provided scientists with a huge array of data for different applications. Artificial intelligence can help refine models used to interpret results, improve understanding, and automate the process.

This project will focus on the Swiss Alps and the change in glacier ice coverage between 2017 and 2023 using Sentinel-2 data. This project will use the NDSI index as a benchmark and for training of the CNN model.

## 1.1 Study Area & Data Collection:

The data for this project has been sourced using the 'Data Collection.ipynb' available in the GitHub repository. The data was collected using the Copernicus database, with the region of interest covering the Swiss Alps, focusing on the Zermatt and Saas-Fe valleys.

The following dates were selected:

 * 15/08/2017
 * 20/07/2023 

 The following satellite data files for the data above were downloaded:

 * S2A_MSIL2A_20170815T102021_N0500_R065_T32TMS_20231005T191904.SAFE
 * S2B_MSIL2A_20230720T101609_N0509_R065_T32TMS_20230720T131906.SAFE

## 1.2 Aims & Objectives

1. To collect Sentinel-2 satellite imagery data from the region of interest in the summer months (2017 & 2023) to ensure cloud-free, comparable data for glacier analysis
2. To calculate the Normalised Difference Snow Index (NDSI) and Normalised Difference Snow and Ice Index (NDSII) as benchmark indicators of snow and glacier-covered areas
3. To apply k-means learning as an unsupervised method for initial classification of areas with glacier ice cover.
4. To train a Convolutional Neural Network (CNN) using labelled data derived from a labelled mask created from the benchmark indicators.
5. To validate the machine learning outputs by comparing them to NDSI/NDSII-based classifications.
6. To quantify changes in glacier extent between 2017 and 2023 by analysing the differences between glacier masks.
7. To evaluate the accuracy and limitations of combining unsupervised and supervised learning for glacier monitoring, including the scalability and potential model improvements
   
# 2.0 Methodology

The flowchart below:

<p align="center">
  <img src="/Images/Glacier_retreat_flowchart.png" width="1000" height="auto"/>
</p>

# 3.0 Results & Discussion

# 4.0 Project enviromental Cost

# References

Sommer, C., Malz, P., Seehaus, T.C. et al. Rapid glacier retreat and downwasting throughout the European Alps in the early 21st century. Nat Commun 11, 3209 (2020). https://doi.org/10.1038/s41467-020-16818-0


