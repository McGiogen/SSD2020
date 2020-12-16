import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64

# Startup config
np.random.seed(550) # for reproducibility

class ForecastResult:
  def __init__(self, dataset, train, forecast, forecast_ci=None):
    self.dataset = dataset
    self.train = train
    self.forecast = forecast
    self.forecast_ci = forecast_ci

class ForecastCi:
  def __init__(self, ciIndex, ciMin, ciMax):
    self.index = ciIndex
    self.max = ciMax
    self.min = ciMin

def print_figure(fig):
	"""
	Converts a figure (as created e.g. with matplotlib or seaborn) to a png image and this
	png subsequently to a base64-string, then prints the resulting string to the console.
	"""
    # The figure will be printed with this format: b'___'
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	print(base64.b64encode(buf.getbuffer()))

def plot_show(shouldShowPlot):
  if (shouldShowPlot == True):
    plt.show()
  else:
    print_figure(plt.gcf())
    plt.clf()

def plot(forecastResult, shouldShowPlot):
  plt.plot(forecastResult.dataset['value'], 'black', label = 'History')
  plt.plot(forecastResult.train,label='Train')
  plt.plot([None for x in forecastResult.train]+[x for x in forecastResult.forecast], label='Forecast')
  #plt.plot([None for x in forecastResult.train]+[None for x in forecastResult.test]+[x for x in forecastResult.forecast], label='Forecast')

  #if forecast_ci != None:
  #  plt.fill_between(forecast_ci.index,
  #                   forecast_ci.iloc[:, 0],
  #                   forecast_ci.iloc[:, 1], color='k', alpha=.25)
  plt.xlabel('time')
  plt.ylabel('value')
  plt.title('Index', color='black')
  plt.legend()

  plot_show(shouldShowPlot)

def autocorrelation(df, months, shouldShowPlot):
  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata) # convert to pandas series

  # acf plot, industrial
  # analisi dei dati con diagramma di autocorrelazione
  import statsmodels.api as sm
  # lags = valori su cui calcolare l'autocorrelazione
  sm.graphics.tsa.plot_acf(data.values, lags=2000)
  plt.title("ACF - Autocorrelation function", color='black')
  plot_show(shouldShowPlot)

def sarima(df, train, forecastSize, shouldShowPlot):
  import pmdarima as pm
  #del df['period']
  model = pm.auto_arima(train, start_p=1, start_q=1, # intervalli validi di p, q, P, Q
                    test='adf', max_p=3, max_q=3, m=1, # stagionalità = 4
                    start_P=0, seasonal=False,
                    d=None, D=0, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True) # False full grid

  print(model.summary()) # stampa i test statistici di affidabilità
  print('LOG Sarima parameters:', model.order)

  # morder = model.order
  # mseasorder = model.seasonal_order

  fitted = model.fit(train)
  yfore = fitted.predict(n_periods=forecastSize) # forecast
  #################### TODO verifica la riga sotto se era valida ####################################
  ypred = fitted.predict_in_sample()[1:]

  return ForecastResult(df, ypred, yfore)

def sarimax(df, train, forecastSize, shouldShowPlot):
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  sarima_model = SARIMAX(train, order=(3,1,1), seasonal_order=(0,0,0,0))
  sfit = sarima_model.fit()

  # Grafico degli errori, istogramma degli error (migliore quando i residui hanno una distribuzione normale), QQ plot, correlogramma (stagionalità sui residui)
  sfit.plot_diagnostics(figsize=(10, 6))
  plot_show(shouldShowPlot)

  # Predizioni in-sample:
  ypred = sfit.predict(start=0,end=train.size)

  # Previsione out-of-sample (che non conosco)
  forewrap = sfit.get_forecast(steps=forecastSize)
  forecast_ci = forewrap.conf_int() # Intervalli di confidenza, più sono ampi e meno affidabile è la previsione
  forecast_val = forewrap.predicted_mean # Valori previsti

  forecast_ci_to_plot = ForecastCi(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1])

  return ForecastResult(df, ypred, forecast_val, forecast_ci_to_plot)

def mlp(df, trainToDiff, forecastSize, shouldShowPlot):
  train = pd.Series(trainToDiff).diff().to_numpy()

  # ------------------------------------------------- neural forecast
  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 250
  generator = TimeseriesGenerator(train, train, length=n_input, batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense

  model = Sequential()
  # lstm_model.add(Dense(30, activation='linear', input_dim=n_input))
  model.add(Dense(250, activation='relu', input_dim=n_input))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  model.fit(generator,epochs=25) #epochs=25
  model.summary()

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  losses_lstm = model.history.history['loss']
  plt.xticks(np.arange(0,21,1)) # convergence trace
  plt.plot(range(len(losses_lstm)),losses_lstm);
  plt.title("Loss")
  plot_show(shouldShowPlot)

  # Prediction
  lstm_forecast = list()
  batch = train[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  for i in range(forecastSize):
    lstm_pred = model.predict(curbatch)[0]
    lstm_forecast.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:],[lstm_pred], axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  yfore = np.transpose(lstm_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"
  yfore = np.r_[[trainToDiff.values[-1]], yfore.cumsum() + trainToDiff.values[-1]] # Transpose: sommo le diff tra loro e aggiungo a tutte il valore di partenza


  # evaluate the keras model
  # accuracy = model.evaluate(generator)
  # print('Accuracy: %.2f' % (accuracy*100))

  # Forecast
  # lstm_forecast_2 = list()
  # batch = np.array(lstm_forecast)[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  # curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  # for i in range(len(test)):
  #   lstm_pred = model.predict(curbatch)[0]
  #   lstm_forecast_2.append(lstm_pred) # Salvo il valore previsto
  #   curbatch = np.append(curbatch[:,1:],[lstm_pred],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  # zfore = np.transpose(lstm_forecast_2).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"

  return ForecastResult(df, trainToDiff, yfore)

def lstm(df, trainToDiff, forecastSize, shouldShowPlot):
  train = pd.Series(trainToDiff).diff().to_numpy()

  # ------------------------------------------------- neural forecast
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaler.fit_transform(train.reshape(-1, 1))
  scaled_train_data = scaler.transform(train.reshape(-1, 1))
  # scaled_test_data = scaler.transform(test.reshape(-1, 1))

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
  lstm_model.fit_generator(generator,epochs=10) #epochs=10

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  losses_lstm = lstm_model.history.history['loss']
  plt.xticks(np.arange(0,21,1)) # convergence trace
  plt.plot(range(len(losses_lstm)),losses_lstm);
  plt.title("Loss")
  plot_show(shouldShowPlot)

  # Prediction
  lstm_predictions_scaled = list()
  batch = scaled_train_data[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input, n_features))  # Creo l'array a partire dalla struttura keras
  for i in range(forecastSize):
    lstm_pred = lstm_model.predict(curbatch)[0]
    lstm_predictions_scaled.append(lstm_pred) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled) # Lo scaler si ricorda la funzione usata per scalare e ricalcola i valori
  yfore = np.transpose(lstm_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"
  yfore = np.r_[[trainToDiff.values[-1]], yfore.cumsum() + trainToDiff.values[-1]] # Transpose: sommo le diff tra loro e aggiungo a tutte il valore di partenza

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

  return ForecastResult(df, train, yfore)

if __name__ == "__main__":
  # change working directory to script path
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  # --- Default values ---
  tec = "mlp"
  months = 12
  shouldShowPlot = False

  # --- Input parameters ---
  print('LOG Number of arguments:', len(sys.argv))
  print('LOG Argument List:', str(sys.argv))
  print('LOG CSV file:', sys.argv[1])

  dffile = sys.argv[1]
  if len(sys.argv) > 2:
    tec = sys.argv[2]
  if len(sys.argv) > 3:
    shouldShowPlot = sys.argv[3] == "show"

  # --- Read data file ---
  df = pd.read_csv("../"+dffile, usecols=[0], names=['value'], header=0)

  # --- Add date to df ---
  #df_date = pd.read_csv("../Data.csv", usecols=[0], names=['date'], header=0)
  #df['period'] = pd.to_datetime(df_date['date'], format="%Y-%m-%d").dt.to_period('D')
  #df.set_index('period')

  # --- Preprocessing data ---
  aValues = df['value'].to_numpy() # array of data
  logdata = np.log(aValues) # log transform
  data = pd.Series(logdata) # convert to pandas series

  # --- Print ACF ---
  from statsmodels.graphics.tsaplots import plot_acf

  # Original Series
  plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
  fig, axes = plt.subplots(2, 2, sharex=True)
  axes[0, 0].plot(df['value']);
  axes[0, 0].set_title('Original Series')
  plot_acf(df['value'], ax=axes[0, 1], lags=2000)

  # 1st Differencing
  axes[1, 0].plot(data);
  #axes[1, 0].set_title('1st Order Differencing')
  axes[1, 0].set_title('Log transformed')
  plot_acf(data.dropna(), ax=axes[1, 1], lags=2000)
  plot_show(shouldShowPlot)

  # --- Train and test set ---
  cutpoint = months * 25;
  train = data[:-cutpoint]
  test = data[-cutpoint:]

  # --- Train, predict and forecast ---
  if tec == "lstm":
    res = lstm(df, train, test.size, shouldShowPlot)
  elif tec == "sarima":
    res = sarima(df, train, test.size, shouldShowPlot)
  elif tec == "sarimax":
    res = sarimax(df, train, test.size, shouldShowPlot)
  else:
    res = mlp(df, train, test.size, shouldShowPlot)

  # recostruction
  #res.train[0] = 0
  #res.forecast[0] = 0
  exptrain = np.exp(res.train) # unlog
  expfore = np.exp(res.forecast)
  #expfore = np.exp(res.forecast+logdata[len(logdata)-1])
  forecast_ci = res.forecast_ci
  #if (forecast_ci != None):
  #  forecast_ci = ForecastCi(forecast_ci.index, np.exp(forecast_ci.max+logdata[len(logdata)-cutpoint-1]), np.exp(forecast_ci.min.cumsum()+logdata[len(logdata)-cutpoint-1]))
  res = ForecastResult(df, exptrain, expfore, forecast_ci)

  plot(res, shouldShowPlot)

  # --- Calculate revenue and risk ---
  actualValue = exptrain[exptrain.size - 1]
  lastMonthValues = res.forecast[-25:]
  forecastAvgValue = np.mean(lastMonthValues)
  accuracy_mape = np.mean(np.abs(lastMonthValues - actualValue)/np.abs(actualValue)) # MAPE

  # --- Value at Risk, historical simulation on the last 12 months ---
  # // TODO TODO TODO confronto con i valori excel TODO TODO TODO
  pctChanges = pd.Series(exptrain[-12*25:]).pct_change().dropna()
  pctChanges.sort_values(inplace=True, ascending=True)
  accuracy_var = pctChanges.quantile(0.05) # VaR con confidenza 95%

  print('LOG Last train value', actualValue)
  print('LOG Last forecasted month (avg)', forecastAvgValue)
  print('REVENUE', forecastAvgValue - actualValue)
  print('REVENUE_PERC', (forecastAvgValue-actualValue)/actualValue)
  print('MAPE', accuracy_mape)
  print('VAR', accuracy_var)
