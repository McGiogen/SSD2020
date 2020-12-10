import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64

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

def plot(forecastResult, shouldShowPlot):
  plt.plot(forecastResult.dataset['value'], 'black', label = 'History')
  plt.plot(forecastResult.train,label='Train')
  plt.plot([None for x in forecastResult.train]+[x for x in forecastResult.test], label='Test')
  plt.plot([None for x in forecastResult.train]+[None for x in forecastResult.test]+[x for x in forecastResult.forecast], label='Forecast')
  plt.xlabel('time')
  plt.ylabel('value')
  plt.title('Index', color='black')
  plt.legend()

  if (shouldShowPlot == True):
    plt.show()
  else:
    print_figure(plt.gcf())

def autocorrelation(df, months, shouldShowPlot):
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
  import pmdarima as pm
  #del df['period']
  model = pm.auto_arima(df.values, start_p=1, start_q=1, # intervalli validi di p, q, P, Q
                    test='adf', max_p=3, max_q=3, m=1, # stagionalità = 4
                    start_P=0, seasonal=False,
                    d=None, D=0, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True) # False full grid

  print(model.summary()) # stampa i test statistici di affidabilità
  morder = model.order
  mseasorder = model.seasonal_order
  fitted = model.fit(df)
  yfore = fitted.predict(n_periods=200) # forecast
  ypred = fitted.predict_in_sample()[1:]
  plot(ForecastResult(df, [], ypred, yfore), True)
  return ForecastResult(df, [], ypred, yfore)

def sarimax(df, months, shouldShowPlot):
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  sarima_model = SARIMAX(df['value'], order=(3,1,1), seasonal_order=(0,0,0,0))
  sfit = sarima_model.fit()
  sfit.plot_diagnostics(figsize=(10, 6))
  plt.show() # Grafico degli errori, istogramma degli error (migliore quando i residui hanno una distribuzione normale), QQ plot, correlogramma (stagionalità sui residui)

  # Predizioni in-sample:
  ypred = sfit.predict(start=0,end=len(df))
  plt.plot(df['value'])
  plt.plot(ypred)
  plt.xlabel('time')
  plt.ylabel('sales')
  plt.show()

  # Previsione out-of-sample (che non conosco)
  forewrap = sfit.get_forecast(steps=200)
  forecast_ci = forewrap.conf_int() # Intervalli di confidenza, più sono ampi e meno affidabile è la previsione
  forecast_val = forewrap.predicted_mean # Valori previsti
  plt.plot(df['value'])
  plt.fill_between(forecast_ci.index,
                  forecast_ci.iloc[:, 0],
                  forecast_ci.iloc[:, 1], color='k', alpha=.25)

  plt.plot(forecast_val)
  plt.xlabel('time');plt.ylabel('values')
  plt.show()

def mlp2(df, months, shouldShowPlot):
  import pandas as pd, numpy as np

  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata).diff() # convert to pandas series and diff transform

  # Preprocessed data
  # plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(data.values);
  plt.show() # data plot

  from statsmodels.graphics.tsaplots import plot_acf

  # Original Series
  plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
  fig, axes = plt.subplots(2, 2, sharex=True)
  axes[0, 0].plot(df['value']);
  axes[0, 0].set_title('Original Series')
  plot_acf(df['value'], ax=axes[0, 1], lags=2000)

  # 1st Differencing
  axes[1, 0].plot(data);
  axes[1, 0].set_title('1st Order Differencing')
  plot_acf(data.dropna(), ax=axes[1, 1], lags=2000)
  plt.show()

  # train and test setm
  train = data[:-200]
  test = data[-200:]
  train[0] = 0

  plt.plot(train.values);
  plt.plot([None for x in train]+test.values.tolist());
  plt.show() # data plot

  # reconstruct = np.exp(np.r_[train,test]) # simple recosntruction

  # ------------------------------------------------- neural forecast
  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 5
  generator = TimeseriesGenerator(train.values, train.values, length=n_input, batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense

  model = Sequential()
  # lstm_model.add(Dense(30, activation='linear', input_dim=n_input))
  model.add(Dense(10, activation='relu', input_dim=n_input, input_shape=()))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  model.fit(generator,epochs=3) #epochs=25
  model.summary()

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  # losses_lstm = model.history.history['loss']
  # plt.xticks(np.arange(0,21,1)) # convergence trace
  # plt.plot(range(len(losses_lstm)),losses_lstm);
  # plt.show()

  # Prediction
  lstm_forecast = list()
  batch = train.values[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  for i in range(len(test)):
    lstm_pred = model.predict(curbatch)[0]
    lstm_forecast.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:],[lstm_pred], axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  yfore = np.transpose(lstm_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"


  # evaluate the keras model
  # accuracy = model.evaluate(generator)
  # print('Accuracy: %.2f' % (accuracy*100))

  # Forecast
  lstm_forecast_2 = list()
  batch = np.array(lstm_forecast)[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  for i in range(len(test)):
    lstm_pred = model.predict(curbatch)[0]
    lstm_forecast_2.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:],[lstm_pred],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  zfore = np.transpose(lstm_forecast_2).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"

  # recostruction
  train[0] = 0
  exptrain = np.exp(train.cumsum()+logdata[0]) # unlog
  #yfore[0] = 0
  exptest = np.exp(yfore.cumsum()+logdata[len(logdata)-201])
  expfore = np.exp(zfore.cumsum()+logdata[len(logdata)-1])
  #expfore = np.exp(zfore)

  plot(ForecastResult(df, exptrain, exptest, expfore), True)
  return ForecastResult(df, exptrain, exptest, expfore)

def mlp(df, months, shouldShowPlot):
  import pandas as pd, numpy as np

  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata).diff() # convert to pandas series and diff transform

  # Preprocessed data
  # plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(data.values);
  plt.show() # data plot

  from statsmodels.graphics.tsaplots import plot_acf

  # Original Series
  plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
  fig, axes = plt.subplots(2, 2, sharex=True)
  axes[0, 0].plot(df['value']);
  axes[0, 0].set_title('Original Series')
  plot_acf(df['value'], ax=axes[0, 1], lags=2000)

  # 1st Differencing
  axes[1, 0].plot(data);
  axes[1, 0].set_title('1st Order Differencing')
  plot_acf(data.dropna(), ax=axes[1, 1], lags=2000)
  plt.show()

  # train and test setm
  train = data[:-200]
  test = data[-200:]
  train[0] = 0

  plt.plot(train.values);
  plt.plot([None for x in train]+test.values.tolist());
  plt.show() # data plot

  # reconstruct = np.exp(np.r_[train,test]) # simple recosntruction

  # ------------------------------------------------- neural forecast
  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 5
  generator = TimeseriesGenerator(train.values, train.values, length=n_input, batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense

  model = Sequential()
  # lstm_model.add(Dense(30, activation='linear', input_dim=n_input))
  model.add(Dense(10, activation='relu', input_dim=n_input))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  model.fit(generator,epochs=3) #epochs=25
  model.summary()

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  # losses_lstm = model.history.history['loss']
  # plt.xticks(np.arange(0,21,1)) # convergence trace
  # plt.plot(range(len(losses_lstm)),losses_lstm);
  # plt.show()

  # Prediction
  lstm_forecast = list()
  batch = train.values[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  for i in range(len(test)):
    lstm_pred = model.predict(curbatch)[0]
    lstm_forecast.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:],[lstm_pred], axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  yfore = np.transpose(lstm_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"


  # evaluate the keras model
  # accuracy = model.evaluate(generator)
  # print('Accuracy: %.2f' % (accuracy*100))

  # Forecast
  lstm_forecast_2 = list()
  batch = np.array(lstm_forecast)[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  for i in range(len(test)):
    lstm_pred = model.predict(curbatch)[0]
    lstm_forecast_2.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:],[lstm_pred],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  zfore = np.transpose(lstm_forecast_2).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"

  # recostruction
  train[0] = 0
  exptrain = np.exp(train.cumsum()+logdata[0]) # unlog
  #yfore[0] = 0
  exptest = np.exp(yfore.cumsum()+logdata[len(logdata)-201])
  expfore = np.exp(zfore.cumsum()+logdata[len(logdata)-1])
  #expfore = np.exp(zfore)

  plot(ForecastResult(df, exptrain, exptest, expfore), True)
  return ForecastResult(df, exptrain, exptest, expfore)

def lstm(df, months, shouldShowPlot):
  import pandas as pd, numpy as np
  import matplotlib.pyplot as plt

  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata) # convert to pandas series

  # Preprocessed data
  # plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(data.values);
  plt.show() # data plot

  # train and test set
  train = data[:-200]
  test = data[-200:]

  # reconstruct = np.exp(np.r_[train,test]) # simple recosntruction

  # ------------------------------------------------- neural forecast
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaler.fit_transform(train.values.reshape(-1, 1))
  scaled_train_data = scaler.transform(train.values.reshape(-1, 1))
  scaled_test_data = scaler.transform(test.values.reshape(-1, 1))

  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 10; n_features = 1
  generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input,
  batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM

  lstm_model = Sequential()
  lstm_model.add(LSTM(10, activation='relu', input_shape=(n_input, n_features), dropout=0.05))
  lstm_model.add(Dense(1))
  lstm_model.compile(optimizer='adam', loss='mse')
  lstm_model.summary()
  lstm_model.fit_generator(generator,epochs=3) #epochs=25

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  losses_lstm = lstm_model.history.history['loss']
  plt.xticks(np.arange(0,21,1)) # convergence trace
  plt.plot(range(len(losses_lstm)),losses_lstm);
  plt.show()

  # Prediction
  lstm_predictions_scaled = list()
  batch = scaled_train_data[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input, n_features))  # Creo l'array a partire dalla struttura keras
  for i in range(len(test)):
    lstm_pred = lstm_model.predict(curbatch)[0]
    lstm_predictions_scaled.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled) # Lo scaler si ricorda la funzione usata per scalare e ricalcola i valori
  yfore = np.transpose(lstm_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"

  # Forecast
  # lstm_forecast_scaled = list()
  # batch = np.array(lstm_predictions_scaled)[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  # curbatch = batch.reshape((1, n_input, n_features))  # Creo l'array a partire dalla struttura keras
  # for i in range(len(test)):
  #   lstm_pred = lstm_model.predict(curbatch)[0]
  #   lstm_forecast_scaled.append(lstm_pred) # Salvo il valore previsto
  #   curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  # lstm_forecast_2 = scaler.inverse_transform(lstm_forecast_scaled) # Lo scaler si ricorda la funzione usata per scalare e ricalcola i valori
  # zfore = np.transpose(lstm_forecast_2).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"

  # recostruction
  exptrain = np.exp(train) # unlog
  exptest = np.exp(yfore)
  expfore = exptest
  #expfore = np.exp(zfore)

  plot(ForecastResult(df, exptrain, exptest, expfore), True)
  return ForecastResult(df, exptrain, exptest, expfore)

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
  #df_date = pd.read_csv("../Data.csv", usecols=[0], names=['date'], header=0)
  #df['period'] = pd.to_datetime(df_date['date'], format="%Y-%m-%d").dt.to_period('D')
  #df.set_index('period')

  #lstm(df, 12, shouldShowPlot)
  if tec == "lstm":
    lstm(df, 12, shouldShowPlot)
  elif tec == "sarima":
    sarima(df, 12, shouldShowPlot)
  elif tec == "sarimax":
    sarimax(df, 12, shouldShowPlot)
  else:
    mlp(df, 12, shouldShowPlot)

  #plt.plot(df)
  #if (shouldShowPlot == True):
  #  plt.show()

  # Finally, print the chart as base64 string to the console.
  # The figure will be printed with this format: b'___'
  #print_figure(plt.gcf())
