# K-Means Clustering Implementation
**Course:** CSCE 474/874: Introduction to Data Mining  
**Team Members:** Brady Lauritsen, Sagun Karki, Max Buettenback, Harrison Johs

## Description
This is a from-scratch implementation of the k-means clustering algorithm designed for continuous variables. The program supports user-defined clusters ($k$), convergence thresholds ($\epsilon$), and iteration limits.

## Installation & Requirements
- Python 3.x
- Standard libraries: `csv`, `math`, `random`, `sys`
- No third-party machine learning libraries (scikit-learn, etc.) are required.

## Usage
Run the script from the terminal using the following syntax:
```bash
python kmeans.py <filename> <k> <epsilon> <max_iterations>