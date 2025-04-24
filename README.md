# GENESIS

Landsat Analysis Toolbox

A comprehensive set of ArcGIS Python tools for processing and analyzing Landsat 8/9 satellite imagery.

Overview
The Landsat Analysis Toolbox provides specialized tools for creating mosaics, calculating spectral indices, performing statistical transformations, and classifying Landsat imagery. This toolbox is designed to streamline remote sensing workflows in ArcGIS Pro for geospatial analysts, environmental scientists, and GIS professionals.

Tools Included
The toolbox contains four main tools:
1. Create Landsat Mosaic
Creates mosaics from Landsat 8/9 scenes with cloud removal and advanced processing options.
Supports temporal filtering by year, month, or season
Includes regional presets for Portugal, Azores, Madeira, Cape Verde, Angola, and Mozambique
Creates geometric median mosaics for improved results
Optional spatial masking

2. Calculate Indices and Composites
Calculates spectral indices and creates band composites for geological, vegetation, and environmental analysis.
Spectral Indices:
Clay Minerals Index (CMI)
Ferrous Minerals Index (FMI)
Iron Oxide Index (IOI)
NDVI (Normalized Difference Vegetation Index)
NDWI (Normalized Difference Water Index)
And many more...

Color Composites:
Natural Color (4,3,2)
False Color (5,4,3)
SWIR Geology Composite (7,6,4)
Clay Minerals Composite (7,5,6)
And others...

3. Statistical Transformations
Performs advanced statistical transformations on Landsat imagery to enhance features and reduce noise.
Minimum Noise Fraction (MNF)
Principal Component Analysis (PCA)
Independent Component Analysis (ICA)

4. Spectral Angle Mapper
Performs Spectral Angle Mapper (SAM) classification using reference spectra from tables, training samples, or individual pixels.
Requirements

ArcGIS Pro 3.0 or higher
Spatial Analyst extension
Image Analyst extension (recommended)
Python 3.x with NumPy, SciPy, and scikit-learn

Installation

Clone this repository or download the ZIP file
Open ArcGIS Pro
Open the Catalog pane
Right-click on Toolboxes and select "Add Toolbox"
Navigate to the downloaded landsat_toolbox.pyt file and select it

Documentation
Each tool includes detailed parameter descriptions and help text within the ArcGIS Pro interface. Additionally, the code contains comprehensive documentation for each function and parameter.
For detailed implementation of statistical algorithms:

The MNF transformation uses a two-step process with noise estimation and signal extraction
PCA implementation follows standard covariance-based computation
ICA uses FastICA algorithm from scikit-learn

Troubleshooting
Common issues and solutions:

Memory Errors: For large datasets, consider processing by tiles or regions
Missing Extensions: Ensure Spatial Analyst extension is properly licensed
Performance Issues: Statistical transformations are computationally intensive; reduce input resolution for testing

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Developed for geological and environmental remote sensing applications
Spectral indices formulations derived from published literature
Transformation algorithms based on established remote sensing methodologies

Contact
For questions or support, please open an issue on this repository.

