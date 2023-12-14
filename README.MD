## ML Models and Data Preprocessing for AI4LS Hackathon

Hackathon - AI4LS Challenge #1: AI for Soil Health

### Soil Data Preprocessing
Scripts for precessing data from the Lucas Soil 2018 dataset are in folder:
- csv_processing

### Training Prediction Models
Prediction models are trained using SVM, Random Forest, and K-mean clustering. Among the approaches, SVM (SVC and SVR) used the most. 
The scripts of model training are organized mainly according to what is the model predicting, the folder are:
- prediction_attributes
- prediction_erosion
- prediction_impacted
- prediction_landuse
- relationship

### Microbiome Data Processing
Only ~30/885 soil samples from the Microbiome Data provided by the Lucas 2018 dataset were processed 
They were processed using the Metabarcoding analysis using the PEMA tool: https://github.com/hariszaf/pema
The processing output V

TU- or an ASV-table with the taxonomies of the taxa found and their abundances in each sample.