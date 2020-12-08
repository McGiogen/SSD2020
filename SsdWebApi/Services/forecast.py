import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64, math

# Startup config
np.random.seed(550) # for reproducibility

class ForecastResult:
  def __init__(self, dataset, train, test, forecast):
    self.dataset = dataset
    self.train = train
    self.test = test
    self.forecast = forecast

def print_figure(fig):
	"""
	Converts a figure (as created e.g. with matplotlib or seaborn) to a png image and this
	png subsequently to a base64-string, then prints the resulting string to the console.
	"""
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	print(base64.b64encode(buf.getbuffer()))

# from series of values to windows matrix
def compute_windows(nparray, npast=1):
  dataX, dataY = [], [] # window and value
  for i in range(len(nparray)-npast-1):
    a = nparray[i:(i+npast), 0]
    dataX.append(a)
    dataY.append(nparray[i + npast, 0])
  return np.array(dataX), np.array(dataY)

def plot(forecastResult, shouldShowPlot):
  plt.plot(forecastResult.dataset)
  plt.plot(np.concatenate((np.full(1,np.nan),forecastResult.train[:,0])))
  plt.plot(np.concatenate((np.full(len(forecastResult.dataset)-len(forecastResult.train)+1,np.nan), forecastResult.test[:,0])))
  plt.plot(np.concatenate((np.full(len(forecastResult.dataset)+1,np.nan), forecastResult.forecast[:,0])))
  plt.xlabel('time')
  plt.ylabel('value')

  if (shouldShowPlot == True):
    plt.show()
  else:
    print_figure(plt.gcf())

def autocorrelation(df, months, shouldShowPlot):
  import pmdarima as pm # pip install pmdarima
  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata) # convert to pandas series

  # acf plot, industrial
  # analisi dei dati con diagramma di autocorrelazione
  import statsmodels.api as sm
  # lags = valori su cui calcolare l'autocorrelazione
  sm.graphics.tsa.plot_acf(data.values, lags=2000)
  plt.show()

def sarima(df, months, shouldShowPlot):
  import pmdarima as pm # pip install pmdarima
  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata) # convert to pandas series

  # plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(data.values)
  plt.show() # data plot


  # acf plot, industrial
  # analisi dei dati con diagramma di autocorrelazione
  import statsmodels.api as sm
  # lags = valori su cui calcolare l'autocorrelazione
  sm.graphics.tsa.plot_acf(data.values, lags=300)
  plt.show()

  # train and test set
  train = data[:-200]
  test = data[-200:]

  # simple reconstruction, not necessary, unused
  # reconstruct = np.exp(np.r_[train,test])


  # auto arima (calcolo dei parametri con una ricerca nello spazio)
  # SARIMA(p,d,q)(P,D,Q,m)
  model = pm.auto_arima(
    train.values, test='adf',
    start_p=1, start_q=1,
    max_p=3, max_q=3, d=None,
    #m=5, D=1, start_P=0,
    #p=2, d=0, q=1,
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
    ) # False full grid

  print(model.summary())
  morder = model.order; print("Sarimax order {0}".format(morder))
  mseasorder = model.seasonal_order;
  print("Sarimax seasonal order {0}".format(mseasorder))

  # predictions and forecasts
  fitted = model.fit(train)
  ypred = fitted.predict_in_sample() # prediction (in-sample)
  yfore = fitted.predict(n_periods=12) # forecast (out-of-sample)
  plt.plot(train.values)
  plt.plot([None for x in range(12)]+[x for x in ypred[12:]])
  plt.plot([None for x in ypred]+[x for x in yfore])
  plt.xlabel('time');plt.ylabel('value')
  plt.show()

  # recostruction
  yplog = pd.Series(ypred)
  expdata = np.exp(yplog) # unlog
  expfore = np.exp(yfore)
  plt.plot([None for x in range(12)]+[x for x in expdata[12:]])
  plt.plot(df["value"])
  plt.plot([None for x in expdata]+[x for x in expfore])
  plt.show()

def mlp(df, months, shouldShowPlot):

  # Keras astrae i modelli e le funzionalità rispetto alle diverse librerie di reti neurali
  from keras.models import Sequential # Percettone multilivello
  from keras.layers import Dense # Livelli con connessioni dense

  dataset = df.values # time series values
  dataset = dataset.astype('float32') # needed for MLP input (richiesto da tensorflow)

  # train - test sets
  cutpoint = int(len(dataset) * 0.7) # 70% train, 30% test
  train, test = dataset[:cutpoint], dataset[cutpoint:]
  print("MAPE Len train={0}, len test={1}".format(len(train), len(test)))

  # sliding window matrices (npast = window width); dim = n - npast - 1
  npast = 7
  # Y = valore desiderato in uscita
  trainX, trainY = compute_windows(train, npast)
  testX, testY = compute_windows(test, npast) # should get also the last npred of train

  # Multilayer Perceptron model
  model = Sequential()
  n_hidden = 8
  n_output = 1
  model.add(Dense(n_hidden, input_dim=npast, activation='relu')) # hidden neurons, add 1 layer
  model.add(Dense(n_output)) # output neurons
  model.compile(loss='mean_squared_error', optimizer='adam') # loss = errore quadratico medio, adam = aggiustamento dei pesi
  model.fit(trainX, trainY, epochs=200, batch_size=10, verbose=2) # batch_size len(trainX)


  # Model performance
  trainScore = model.evaluate(trainX, trainY, verbose=0)
  print('MAPE Score on train: MSE = {0:0.2f} '.format(trainScore))
  testScore = model.evaluate(testX, testY, verbose=0)
  print('MAPE Score on test: MSE = {0:0.2f} '.format(testScore))

  trainPredict = model.predict(trainX) # predictions
  testForecast = model.predict(testX) # forecast

  #return ForecastResult(dataset, trainPredict, testForecast, )
  # plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(dataset)
  plt.plot(np.concatenate((np.full(1,np.nan),trainPredict[:,0])))
  plt.plot(np.concatenate((np.full(len(train)+1,np.nan), testForecast[:,0])))
  plt.xlabel('time');plt.ylabel('value')

  if (shouldShowPlot == True):
    plt.show()
  print_figure(plt.gcf())

def lstm(df, months, shouldShowPlot):
  import pandas as pd, numpy as np, os
  import matplotlib.pyplot as plt

  df.set_index('period')
  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata) # convert to pandas series

  plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(data.values);
  plt.show() # data plot

  # train and test set
  train = data[:-200]
  test = data[-200:]

  reconstruct = np.exp(np.r_[train,test]) # simple recosntruction

  # ------------------------------------------------- neural forecast
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaler.fit_transform(train.values.reshape(-1, 1))
  scaled_train_data = scaler.transform(train.values.reshape(-1, 1))
  scaled_test_data = scaler.transform(test.values.reshape(-1, 1))

  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 12; n_features = 1
  generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input,
  batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM

  lstm_model = Sequential()
  lstm_model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features), dropout=0.05))
  lstm_model.add(Dense(1))
  lstm_model.compile(optimizer='adam', loss='mse')
  lstm_model.summary()
  lstm_model.fit_generator(generator,epochs=25)

  losses_lstm = lstm_model.history.history['loss']
  plt.xticks(np.arange(0,21,1)) # convergence trace
  plt.plot(range(len(losses_lstm)),losses_lstm);
  plt.show()

  lstm_predictions_scaled = list()
  batch = scaled_train_data[-n_input:]
  curbatch = batch.reshape((1, n_input, n_features))
  for i in range(len(test)):
    lstm_pred = lstm_model.predict(curbatch)[0]
    lstm_predictions_scaled.append(lstm_pred)
    curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1)

  lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)
  yfore = np.transpose(lstm_forecast).squeeze()

  # recostruction
  expdata = np.exp(train) # unlog
  expfore = np.exp(yfore)
  plt.plot(df["value"], label="values")
  plt.plot(expdata,label='expdata')
  plt.plot([None for x in expdata]+[x for x in expfore], label='forecast')
  plt.legend()
  plt.show()

if __name__ == "__main__":
  # change working directory to script path
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  print('MAPE Number of arguments:', len(sys.argv))
  print('MAPE Argument List:', str(sys.argv), ' first true arg:',sys.argv[1])

  tec = sys.argv[2] if len(sys. argv) >= 2 else "mlp"
  shouldShowPlot = len(sys. argv) >= 3 and sys.argv[3] == "show"

  dffile = sys.argv[1]
  df = pd.read_csv("../"+dffile, usecols=[0], names=['value'], header=0)

  # Add date to df
  df_date = pd.read_csv("../Data.csv", usecols=[0], names=['date'], header=0)
  #df['period'] = pd.to_datetime(df_date['date'], format="%Y-%m-%d").dt.to_period('D')

  if tec == "sarima":
    sarima(df, 12, shouldShowPlot)
  elif tec == "autocorrelation":
    autocorrelation(df, 12, shouldShowPlot)
  elif tec == "lstm":
    lstm(df, 12, shouldShowPlot)
  else:
    mlp(df, 12, shouldShowPlot)

  #plt.plot(df)
  #if (shouldShowPlot == True):
  #  plt.show()

  # Finally, print the chart as base64 string to the console.
  # The figure will be printed with this format: b'___'
  #print_figure(plt.gcf())
