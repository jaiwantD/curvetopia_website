from flask import Flask, request, render_template, redirect, url_for, send_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Helper function to calculate distances from the center
def calc_R(xc, yc, x, y):
    return np.sqrt((x - xc)**2 + (y - yc)**2)

# Function to optimize
def f_2(c, x, y):
    Ri = calc_R(*c, x, y)
    return Ri - Ri.mean()

# Function to detect circles
def detect_circle(coords):
    x, y = coords[:, 0], coords[:, 1]
    x_m, y_m = np.mean(x), np.mean(y)
    initial_guess = [x_m, y_m]
    res = least_squares(f_2, initial_guess, args=(x, y))
    xc, yc = res.x
    Ri = calc_R(xc, yc, x, y)
    R = Ri.mean()
    return xc, yc, R

# Function to detect lines
def detect_line(coords):
    x, y = coords[:, 0], coords[:, 1]
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model.coef_[0], model.intercept_

# Function to plot detected shapes
def plot_shapes(coordinates, labels, circle_params):
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    
    def circles_overlap(c1, c2):
        if c1[0] is None or c2[0] is None:
            return False
        dist_centers = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return dist_centers < (c1[2] + c2[2])
    
    for i, coords in enumerate(coordinates):
        x, y = coords[:, 0], coords[:, 1]
        label = labels[i]
        if label == 'line':
            coef, intercept = detect_line(coords)
            plt.plot(x, coef * x + intercept, color=colors[i % len(colors)], label=f'Line {i+1}')
        elif label == 'circle':
            xc, yc, R = circle_params[i]
            overlap = False
            for j in range(i):
                if circles_overlap(circle_params[i], circle_params[j]):
                    overlap = True
                    break
            if not overlap:
                circle = plt.Circle((xc, yc), R, color=colors[i % len(colors)], fill=False, label=f'Circle {i+1}')
                plt.gca().add_artist(circle)
        plt.scatter(x, y, color=colors[i % len(colors)], s=10, label=f'Points {i+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# Function to read CSV and parse paths and coordinates
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            paths_XYs = read_csv(file_path)
            coordinates = [item for sublist in paths_XYs for item in sublist]
            labels = []
            circle_params = []

            for coords in coordinates:
                if len(coords) < 3:
                    labels.append('line')
                    circle_params.append((None, None, None))
                    continue

                coef, intercept = detect_line(coords)
                residuals = np.abs(coords[:, 1] - (coef * coords[:, 0] + intercept))
                if np.max(residuals) < 1e-2:
                    labels.append('line')
                    circle_params.append((None, None, None))
                else:
                    xc, yc, R = detect_circle(coords)
                    Ri = calc_R(xc, yc, coords[:, 0], coords[:, 1])
                    std_dev = np.std(Ri)
                    arc_length = np.sum(np.sqrt(np.diff(coords[:, 0])**2 + np.diff(coords[:, 1])**2))

                    if R > arc_length:
                        labels.append('line')
                        circle_params.append((None, None, None))
                    else:
                        if std_dev < 0.1 * R:
                            labels.append('circle')
                            circle_params.append((xc, yc, R))
                        else:
                            labels.append('line')
                            circle_params.append((None, None, None))

            img = plot_shapes(coordinates, labels, circle_params)
            return send_file(img, mimetype='image/png')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
