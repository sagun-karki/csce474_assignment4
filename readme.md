# K-Means Clustering Implementation
**Course:** CSCE 474/874: Introduction to Data Mining  
**Team Members:** Brady Lauritsen, Sagun Karki, Max Buettenback, Harrison Johs

## Description
This is a from-scratch implementation of the k-means clustering algorithm designed for continuous variables. The program supports user-defined clusters ($k$), convergence thresholds ($\epsilon$), and iteration limits.

### Files
1. [src/kmeans.py](src/kmeans.py): Contains the k-means algorithm implementation.
2. [src/results](src/results/): Contains the results plots.
3. [src/Lauritsen_Karki_Buettenback_Johs.pdf](src/Lauritsen_Karki_Buettenback_Johs.pdf): Contains the report along side the python and weka outputs and comparison.

## Installation & Requirements
- Python 3.x
- Standard libraries: `csv`, `math`, `random`, `sys`
- `matplotlib` for plotting.

## Usage
Run the script from the terminal using the following syntax:
```bash
python kmeans.py <filename> <k> <epsilon> <max_iterations>
```