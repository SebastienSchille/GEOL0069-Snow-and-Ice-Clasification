<p align="center">
  <img src="/Images/ndsi_deployment_region.png" width="600" height="auto"/>
</p>

# Monitoring Glacier Retreat in the Swiss Alps Using Supervised and Unsupervised Machine Learning.

# 1.0 Project Background

Glaciers are extremely sensitive to climate change and can be studied to highlight the rapid change in alpine landscape in the 21st Century (Sommer et al., 2020). There has been an alarming increase in glacier retreat around the globe, leading to catastrophic consequences on mountain ecosystems, freshwater supplies, tourism, and local economies. Accurate monitoring is not only crucial for assessing the impacts of climate change but also for informing mitigation and adaptation strategies.

Traditional data collection methods, such as sample collection and manual interpretation of satellite data, can be highly time-consuming and often have limited insights. Earth observation satellites launched within the Copernicus program have provided scientists with various datasets for a multitude of applications. Artificial intelligence can help refine models used to interpret results, improve understanding, and automate the process.

This project will focus on the Swiss Alps and the change in glacier ice coverage between 2017 and 2023 using Sentinel-2 data. The Normalised Difference Snow Index (NDSI) will be used as a benchmark and for training the CNN model. Unsupervised and supervised AI models will be deployed, with results quantified and discussed. Additionally, this project hopes to demonstrate proof of concept with a model that is transferable to other regions and discuss the limitations and recommended model changes.

## 1.1 Study Area & Data Collection

The data for this project has been sourced using the 'Data Collection.ipynb' available in the GitHub repository. The data was collected using the Copernicus database, with the region of interest covering the Swiss Alps, focusing on the Zermatt and Saas-Fe valleys.

The following dates were selected:

 * 15/08/2017
 * 20/07/2023 

 The following satellite data files for the data above were downloaded:

 * S2A_MSIL2A_20170815T102021_N0500_R065_T32TMS_20231005T191904.SAFE
 * S2B_MSIL2A_20230720T101609_N0509_R065_T32TMS_20230720T131906.SAFE

The following area were selected:

<p align="center">
  <img src="/Images/area_of_interest.png" width="1000" height="auto"/>
</p>

## 1.2 Aims & Objectives

1. To collect Sentinel-2 satellite imagery data from the region of interest in the summer months (2017 & 2023) to ensure cloud-free, comparable data for glacier analysis
2. To calculate the Normalised Difference Snow Index (NDSI) and Normalised Difference Snow and Ice Index (NDSII) as benchmark indicators of snow and glacier-covered areas
3. To apply k-means learning as an unsupervised method for initial classification of areas with glacier ice cover.
4. To train a Convolutional Neural Network (CNN) using labelled data derived from a labelled mask created from the benchmark indicators.
5. To validate the machine learning outputs by comparing them to NDSI/NDSII-based classifications.
6. To quantify changes in glacier extent between 2017 and 2023 by analysing the differences between glacier masks.
7. To evaluate the accuracy and limitations of combining unsupervised and supervised learning for glacier monitoring, including the scalability and potential model improvements
   
# 2.0 Methodology

Following data collecction and visulasation the test, validation and deployment area were selected. The flowchart below illustrates the work flow of the project:

<p align="center">
  <img src="/Images/Glacier_retreat_flowchart.png" width="1000" height="auto"/>
</p>

## 2.1 Sentinel-2 Optical sensors

Sentinel-2 is equipptd with a with an advanced MultiSpectral Instrument (MSI). This sensor captures high-resolution imagery across 13 spectral bands, specifically designed for earth observation applications. This project will use the bands from the table below:

| Band | Name                | Wavelength (nm) | Resolution | Use for Glacier Mapping                           |
|------|---------------------|------------------|-------------|---------------------------------------------------|
| B2   | Blue                | 490              | 10 m        | Atmospheric correction, snow detection            |
| B3   | Green               | 560              | 10 m        | Used in **NDSI** to detect snow/ice               |
| B4   | Red                 | 665              | 10 m        | General land cover classification                 |
| B8A  | Narrow NIR          | 865              | 20 m        | Used in **NDSII**, better glacier detection       |
| B11  | SWIR                | 1610             | 20 m        | Used in **NDSI & NDSII**, ice and snow absorption |

## 2.2 Snow and Ice Indices

This project uses a combination of indices that have been developed for snow and ice detection. The Normalised Difference Snow Index (NDSI) is used to identify snow-covered areas by using the green band and the SWIR reflectance. Snow and ice reflects green visable light while it is highly absoarbative of short wave infrared (SWIR). Aditionally clouds reflect the SWIR band enabling to disregard them as snow or ice. The NDSI can be caluclated from the equation below:

**NDSI** = (B3 − B11) / (B3 + B11)  
*Where:*  
- **B3** = Green band (~560 nm)  
- **B11** = Short-Wave Infrared (SWIR) band (~1610 nm)

<p align="center">
  <img src="/Images/NDSII_calculation.png" width="1000" height="auto"/>
</p>

However, this index will struggle to distinguish between snow and ice, leading to misidentification. Short snowfall event could lead to altering results when believed to be glacier ice. To combat this issue the Normalised Difference Snow and Ice Index (NDSII) can be used. This index helps to differenciate between snow and ice by using the NIR band. Glacier ice will reflect the NIR band more than snow hence aiding to distinguish them.

**NDSII** = (B11 − B8A) / (B11 + B8A)  
*Where:*  
- **B11** = Short-Wave Infrared (SWIR) band (~1610 nm)  
- **B8A** = Narrow Near-Infrared (NIR) band (~865 nm)

<p align="center">
  <img src="/Images/NDSI_calculation.png" width="1000" height="auto"/>
</p>

---

The combination of these two indices was used to create a benchmark mask.
The threshold is set as: glacier_mask = (ndsi > 0.4) & (ndsii > 0.3).

| **Aspect**       | **NDSI (> 0.4)**                              | **NDSII (> 0.3)**                           | **Overall Effect**                                                  |
|------------------|----------------------------------------------|----------------------------------------------|---------------------------------------------------------------------|
| What it detects  | Clean snow/ice (high green, low SWIR)        | Snow/ice with high NIR, low SWIR             | Targets clean glacier surfaces with high reflectance in green/NIR  |
| Typical use      | Glacier/snow detection, conservative cutoff  | Snow/ice vs water or wet rock discrimination | Helps refine glacier detection beyond NDSI alone                   |
| Sensitivity      | May miss debris-covered glaciers             | May exclude moist or shadowed glacier areas  | Might underdetect complex surfaces like dirty ice or thin snow     |

<p align="center">
  <img src="/Images/NDSI_NDSII_mask.png" width="1000" height="auto"/>
</p>

## 2.3 K-means learning

K-means classification is an unsupervised machine learning tool that doesn't require labelled data. It will compare neighbouring pixels to find trends, patterns and similarity. The initialisation, location, and parameters of the centroids used in the model can be changed to meet the user's needs. The model can be adjusted to change the number of clusters or groups to be identified. In this project, this method was used as a baseline classifier to find ice-covered areas.


<p align="center">
  <img src="/Images/K_means_mask.png" width="1000" height="auto"/>
</p>


<p align="center">
  <img src="/Images/K_means_confusion_matrix.png" width="500" height="auto"/>
</p>


## 2.4 CNN model

Convolutional Neural Networks (CNNs) are a type supervised machine learning tool trained by using labelled data. These networks are designed to work with classification tasks such as imagery or object detection. They use a gridlike topogrpahy to detect patterns, trends and texture. In this project a CNN model will be trained using labelled data from the benchmark indices and the bands listed in table X.


# 3.0 Results & Discussion


<p align="center">
  <img src="/Images/CNN_validation_region.png" width="1000" height="auto"/>
</p>


<p align="center">
  <img src="/Images/CNN_deployment_region.png" width="1000" height="auto"/>
</p>



Limitations:
- Water and cloud detection

# 4.0 Project enviromental Cost

# References

Sommer, C., Malz, P., Seehaus, T.C. et al. Rapid glacier retreat and downwasting throughout the European Alps in the early 21st century. Nat Commun 11, 3209 (2020). https://doi.org/10.1038/s41467-020-16818-0


