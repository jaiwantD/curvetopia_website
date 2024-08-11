# Shape Detection and Regularization

This project is focused on detecting and regularizing freehand drawn shapes using coordinate data. The core functionalities include detecting circles and lines from a set of coordinates, optimizing these detections, and plotting the identified shapes. The implementation uses mathematical optimization techniques and machine learning models to achieve this.

## Table of Contents
- [Overview](#overview)
- [Mathematical Concepts](#mathematical-concepts)
- [Machine Learning Models](#machine-learning-models)
- [Code Explanation](#code-explanation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Overview
This project uses a combination of mathematical optimization and machine learning techniques to detect and regularize shapes such as circles and lines from a set of coordinates.

## Mathematical Concepts

### 1. Circle Detection
- **Center and Radius Calculation:** The center \((x_c, y_c)\) and radius \(R\) of the circle are calculated by minimizing the difference between the distance of each point from the center and the mean distance (radius).
- **Least Squares Optimization:** The function `f_2` is defined as the difference between each radius \(R_i\) and the mean radius \(R\), and is minimized using a least squares approach.
  - [Least Squares Method Explained](https://www.statology.org/least-squares-method/)
  - [Circle Fitting Using Least Squares](https://people.cas.uab.edu/~mosya/cl/cl.html)

### 2. Line Detection
For line detection:
- **Linear Regression:** The points are fit to a straight line \(y = mx + b\) using linear regression. This involves finding the slope \(m\) and intercept \(b\) that minimize the sum of squared differences between the actual points and the predicted points on the line.
  - [Linear Regression - A Complete Introduction](https://towardsdatascience.com/a-complete-introduction-to-linear-regression-als

### 3. Overlap Handling for Circles
- **Circle Overlap:** To avoid overlapping circles, the distance between the centers of the circles is calculated. If this distance is less than the sum of the radii of the two circles, they are considered overlapping, and only one of them is plotted.
  - [Circle Overlap in Geometry](https://math.stackexchange.com/questions/104682/when-do-two-circles-overlap)

### 4. Arc Length Consideration
- **Arc Length:** To distinguish between a full circle and an arc or partial circle, the arc length of the points is compared with the calculated radius. If the radius is greater than the arc length, the points are more likely to form a line rather than a circle.
  - [Arc Length Formula](https://www.cuemath.com/geometry/arc-length/)

## Machine Learning Models

### Linear Regression
The code uses `LinearRegression` from scikit-learn to detect lines. Linear regression is a supervised learning algorithm that models the relationship between a dependent variable \(y\) and one or more independent variables \(x\). In this case, the independent variable is the x-coordinate, and the dependent variable is the y-coordinate of the points.

The model fits a line \(y = mx + b\) that best represents the data by minimizing the sum of squared residuals (differences between the actual y-values and the predicted y-values).
  - [Linear Regression in Scikit-Learn](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

## Code Explanation

### 1. Circle Detection (`detect_circle`)
The `detect_circle` function calculates the center and radius of a circle using the least squares optimization technique. It takes the x and y coordinates of the points, computes an initial guess for the center as the mean of the points, and then refines this guess by minimizing the variance of the radii.

### 2. Line Detection (`detect_line`)
The `detect_line` function fits a line to the points using linear regression. It returns the slope and intercept of the line.

### 3. Shape Plotting (`plot_shapes`)
The `plot_shapes` function visualizes the detected shapes. It handles multiple shapes and checks for overlapping circles to ensure a clean plot.

### 4. CSV Parsing (`read_csv`)
The `read_csv` function reads the coordinate data from a CSV file, organizes it into separate paths, and returns the coordinates in a structured format.

### 5. Main Detection Loop
The loop iterates through the paths, classifying each set of points as either a line or a circle based on residuals and standard deviations. It then calls the appropriate detection function and stores the results for plotting.

## Usage

To use this code:

1. Prepare a CSV file with coordinate data. Each path should be represented with a unique identifier in the first column.
2. Update the `csv_path` variable with the path to your CSV file.
3. Run the script to detect and plot the shapes.

## Dependencies

The following Python packages are required:
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`

You can install these dependencies using `pip`:

```bash
pip install numpy matplotlib scipy scikit-learn

The speciality is the project doesn't use the neural network or diffusion models. It's completely depends mathematical and statistical analysis.


