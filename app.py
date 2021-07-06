import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['mat', 'csv'])
model = tf.keras.models.load_model('mymodel.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(file.__dir__())
            filename1 = os.path.splitext(filename)[0]
            print(filename1)
            cycle_1 = int(request.form['Current_Cycle'])
            mat = loadmat(UPLOAD_FOLDER + "/" + filename1 + '.mat')
            print('Total data in dataset: ', len(mat[filename1][0, 0]['cycle'][0]))
            counter = 0
            dataset = []
            capacity_data = []

            for i in range(len(mat[filename1][0, 0]['cycle'][0])):
                row = mat[filename1][0, 0]['cycle'][0, i]
                if row['type'][0] == 'discharge':
                    ambient_temperature = row['ambient_temperature'][0][0]
                    date_time = datetime.datetime(int(row['time'][0][0]),
                                                  int(row['time'][0][1]),
                                                  int(row['time'][0][2]),
                                                  int(row['time'][0][3]),
                                                  int(row['time'][0][4])) + datetime.timedelta(
                        seconds=int(row['time'][0][5]))
                    data = row['data']
                    capacity = data[0][0]['Capacity'][0][0]
                    for j in range(len(data[0][0]['Voltage_measured'][0])):
                        voltage_measured = data[0][0]['Voltage_measured'][0][j]
                        current_measured = data[0][0]['Current_measured'][0][j]
                        temperature_measured = data[0][0]['Temperature_measured'][0][j]
                        current_load = data[0][0]['Current_load'][0][j]
                        voltage_load = data[0][0]['Voltage_load'][0][j]
                        time = data[0][0]['Time'][0][j]
                        dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                        voltage_measured, current_measured,
                                        temperature_measured, current_load,
                                        voltage_load, time])
                    capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
                    counter = counter + 1
            dataset, capacity = [pd.DataFrame(data=dataset, columns=['cycle', 'ambient_temperature', 'datetime',
                                                                     'capacity', 'voltage_measured',
                                                                     'current_measured', 'temperature_measured',
                                                                     'current_load', 'voltage_load', 'time']),
                                 pd.DataFrame(data=capacity_data,
                                              columns=['cycle', 'ambient_temperature', 'datetime', 'capacity'])]
            attrib = ['cycle', 'datetime', 'capacity']
            dis_ele = capacity[attrib]
            rows = ['cycle', 'capacity']
            dataset = dis_ele[rows]
            data_train = dataset[(dataset['cycle'] < cycle_1)]
            data_set_train = data_train.iloc[:, 1:2].values
            data_test = dataset[(dataset['cycle'] >= cycle_1)]
            data_set_test = data_test.iloc[:, 1:2].values

            sc = MinMaxScaler(feature_range=(0, 1))
            data_set_train = sc.fit_transform(data_set_train)
            data_set_test = sc.transform(data_set_test)
            input1 = cycle_1 - 1
            X_train = []
            y_train = []
            # take the last 10t to predict 10t+1
            for i in range(10,input1):
                X_train.append(data_set_train[i - 10:i, 0])
                y_train.append(data_set_train[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            data_total = pd.concat((data_train['capacity'], data_test['capacity']), axis=0)
            inputs = data_total[len(data_total) - len(data_test) - 10:].values
            print(len(data_total) - len(data_test) - 10)
            inputs = inputs.reshape(-1, 1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(10, len(data_test)+10):
                X_test.append(inputs[i - 10:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            pred = model.predict(X_test)
            print(pred.shape)
            pred = sc.inverse_transform(pred)
            pred = pred[:, 0]
            tests = data_test.iloc[:, 1:2]
            rmse = np.sqrt(mean_squared_error(tests, pred))
            print('Test RMSE: %.3f' % rmse)
            metrics.r2_score(tests, pred)

            ln = len(data_train)
            data_test['pre'] = pred
            plot_df = dataset.loc[(dataset['cycle'] >= 1), ['cycle', 'capacity']]
            plot_per = data_test.loc[(data_test['cycle'] >= ln), ['cycle', 'pre']]
            plt.figure(figsize=(16, 10))
            plt.plot(plot_per['cycle'], plot_per['pre'], label="Prediction data", color='red')
            plt.plot([0., 168], [1.4, 1.4], dashes=[6, 2], label="treshold")
            plt.ylabel('Capacity')
            adf = plt.gca().get_xaxis().get_major_formatter()
            plt.xlabel('cycle')
            plt.legend()
            plt.title('RUL Prediction For Battery ' + filename1 + 'Starting in ' + str(cycle_1) + 'Window_Size = 10')
            plt.savefig('static/trial.png')
            Pfil = 0
            a = data_test['capacity'].values
            b = data_test['pre'].values
            for i in range(len(a)):
                pred = b[i]
                if pred < 1.4:
                    k = i
                    Pfil = k
                    break
            print(Pfil)
            print("The prediction fail at cycle number: " + str(Pfil + ln - cycle_1))
            return render_template('complete.html', prediction_text='Cycle Left Till Degradation   ' + '' + str(Pfil + ln - cycle_1),pointPfil = (Pfil+ln - cycle_1))
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
