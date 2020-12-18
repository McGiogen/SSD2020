import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64
from statsmodels.graphics.tsaplots import plot_acf

# Startup config
np.random.seed(550) # for reproducibility

class ForecastResult:
  def __init__(self, train, forecast, forecast_ci=None, dataset=None):
    self.train = train
    self.forecast = forecast
    self.forecast_ci = forecast_ci
    self.dataset = dataset

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
  plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plt.plot(forecastResult.dataset['value'], 'black', label = 'History')
  plt.plot(forecastResult.train, 'blue', label='Prediction')
  # A fini grafici aggiungo anche l'ultimo valore del train-set all'inizio della serie
  plt.plot([None for x in forecastResult.train[:-1]]+[forecastResult.dataset['value'][forecastResult.train.size-1]]+[x for x in forecastResult.forecast], 'red', label='Forecast')

  if forecast_ci != None:
    plt.fill_between(forecast_ci.index,
                     forecast_ci.max,
                     forecast_ci.min, color='k', alpha=.25)
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
  sm.graphics.tsa.plot_acf(data.values, lags=600)
  plt.title("ACF - Autocorrelation function", color='black')
  plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plot_show(shouldShowPlot)

def sarima(train, forecastSize, shouldShowPlot):
  import pmdarima as pm
  model = pm.auto_arima(train, start_p=1, start_q=1, # intervalli validi di p, q, P, Q
                    test='adf', max_p=3, max_q=3, m=1, # m = stagionalità
                    start_P=0, seasonal=False,
                    d=None, D=0, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True) # False full grid

  print(model.summary()) # stampa i test statistici di affidabilità
  print('LOG Sarima parameters:', model.order, model.seasonal_order)

  fitted = model.fit(train)

  # Predizioni in-sample
  ypred = fitted.predict_in_sample()[1:] # Rimuove il primo valore perché zero

  # Previsione out-of-sample
  yfore = fitted.predict(n_periods=forecastSize)

  return ForecastResult(ypred, yfore)

def sarimax(train, forecastSize, shouldShowPlot):
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  sarima_model = SARIMAX(train, order=(2,1,0), seasonal_order=(0,0,0,0))
  sfit = sarima_model.fit()

  # Grafico degli errori, istogramma degli error (migliore quando i residui hanno una distribuzione normale), QQ plot, correlogramma (stagionalità sui residui)
  sfit.plot_diagnostics(figsize=(10, 8))
  plot_show(shouldShowPlot)

  # Predizioni in-sample
  ypred = sfit.predict(start=0,end=train.size)[1:] # Rimuove il primo valore perché zero

  # Previsione out-of-sample
  forewrap = sfit.get_forecast(steps=forecastSize)
  forecast_ci = forewrap.conf_int() # Intervalli di confidenza, più sono ampi e meno affidabile è la previsione
  forecast_val = forewrap.predicted_mean # Valori previsti

  forecast_ci_to_plot = ForecastCi(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1])

  return ForecastResult(ypred, forecast_val, forecast_ci_to_plot)

def mlp(trainToDiff, forecastSize, shouldShowPlot):
  # --- Diff transform ---
  train = trainToDiff.diff().dropna().to_numpy()

  # --- Print ACF ---

  # Original Series
  plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120})
  fig, axes = plt.subplots(1, 2, sharex=False)
  axes[0].plot(train)
  axes[0].set_title('1st Order differencing (train data)')
  plot_acf(train, ax=axes[1], lags=50)
  plot_show(shouldShowPlot)

  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 30
  generator = TimeseriesGenerator(train, train, length=n_input, batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense

  # --- Generazione modello ---
  # Dimensione training set: 5223 - 300 = 4923
  # Pesi: neuroniInput * neuroniNascosti + neuroniNascosti*neuroniOutput
  #       30*15 + 15*15 + 15*15 + 25*1 = 450 + 225 + 225 + 15 = 915
  model = Sequential()
  model.add(Dense(15, activation='relu', input_dim=n_input))
  model.add(Dense(15, activation='relu', input_dim=15))
  model.add(Dense(15, activation='relu', input_dim=15))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  model.fit(generator,epochs=25)
  model.summary()

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  losses_mlp= model.history.history['loss']
  plt.xticks(np.arange(0,21,1)) # convergence trace
  plt.plot(range(len(losses_mlp)),losses_mlp)
  plt.title("Loss")
  plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plot_show(shouldShowPlot)

  # Predizioni in-sample
  #mlp_predict = model.predict(generator)

  #ypred = np.transpose(mlp_predict).squeeze()
  #ypred = ypred.cumsum() + trainToDiff.values[n_input-1]
  ypred = [0 for x in trainToDiff]

  # Forecast
  mlp_forecast = list()
  batch = train[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input))  # Creo l'array a partire dalla struttura keras
  for i in range(forecastSize):
    mlp_fore = model.predict(curbatch)[0]
    mlp_forecast.append(mlp_fore) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:],[mlp_fore], axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  yfore = np.transpose(mlp_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"
  yfore = yfore.cumsum() + trainToDiff.values[-1] # Sommo le diff tra loro e aggiungo a tutte il valore di partenza

  return ForecastResult(ypred, yfore)

def lstm(trainToDiff, forecastSize, shouldShowPlot):
  # --- Diff transform ---
  train = trainToDiff.diff().dropna().to_numpy()

  # --- Print ACF ---

  # Original Series
  plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120})
  fig, axes = plt.subplots(1, 2, sharex=False)
  axes[0].plot(train)
  axes[0].set_title('1st Order differencing (train data)')
  plot_acf(train, ax=axes[1], lags=50)
  plot_show(shouldShowPlot)

  # --- Scaling tranform ---
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaler.fit_transform(train.reshape(-1, 1))
  scaled_train_data = scaler.transform(train.reshape(-1, 1))
  # scaled_test_data = scaler.transform(test.reshape(-1, 1))

  from keras.preprocessing.sequence import TimeseriesGenerator
  n_input = 30; n_features = 1
  generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input,
  batch_size=1)

  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM

  # --- Generazione modello ---
  # Dimensione training set: 5223 - 300 = 4923
  # Pesi: neuroniInput * neuroniNascosti + neuroniNascosti*neuroniOutput
  #       30*15 + 15*1 = 450 + 15 = 465
  lstm_model = Sequential()
  lstm_model.add(LSTM(15, activation='relu', input_shape=(n_input, n_features), dropout=0.05))
  lstm_model.add(Dense(1))
  lstm_model.compile(optimizer='adam', loss='mse')
  lstm_model.summary()
  lstm_model.fit_generator(generator,epochs=10)

  # Andamento loss
  # Ogni volta lavoro su un sottoinsieme di dati quindi può peggiorare
  losses_lstm = lstm_model.history.history['loss']
  plt.xticks(np.arange(0,21,1)) # convergence trace
  plt.plot(range(len(losses_lstm)),losses_lstm)
  plt.title("Loss")
  plt.rcParams["figure.figsize"] = (10,8) # redefines figure size
  plot_show(shouldShowPlot)

  # Predizioni in-sample
  #lstm_predict_scaled = lstm_model.predict(generator)

  #lstm_predict = scaler.inverse_transform(lstm_predict_scaled)
  #ypred = np.transpose(lstm_predict).squeeze()
  #ypred = ypred.cumsum() + trainToDiff.values[n_input-1]
  ypred = [0 for x in trainToDiff]

  # Forecast
  lstm_forecast_scaled = list()
  batch = scaled_train_data[-n_input:]  # Metto dentro gli ultimi dati che userò come input per il primo valore previsto
  curbatch = batch.reshape((1, n_input, n_features))  # Creo l'array a partire dalla struttura keras
  for i in range(forecastSize):
    lstm_fore = lstm_model.predict(curbatch)[0]
    lstm_forecast_scaled.append(lstm_fore) # Salvo il valore previsto
    curbatch = np.append(curbatch[:,1:,:],[[lstm_fore]],axis=1) # Rimuovo il valore più vecchio e aggiungo quello appena previsto

  lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled) # Lo scaler si ricorda la funzione usata per scalare e ricalcola i valori
  yfore = np.transpose(lstm_forecast).squeeze() # Transpose: da array verticale lo faccio diventare un array orizzontale/"normale"
  yfore = yfore.cumsum() + trainToDiff.values[-1] # Sommo le diff tra loro e aggiungo a tutte il valore di partenza

  return ForecastResult(ypred, yfore)

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

  # Original Series
  plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':120})
  fig, axes = plt.subplots(2, 2, sharex=False)
  axes[0, 0].plot(df['value']);
  axes[0, 0].set_title('Original Series')
  plot_acf(df['value'], ax=axes[0, 1], lags=50)

  # Transformed Series
  axes[1, 0].plot(data);
  axes[1, 0].set_title('Log transformed')
  plot_acf(data.dropna(), ax=axes[1, 1], lags=50)
  plot_show(shouldShowPlot)

  # --- Train and test set ---
  cutpoint = months * 25;
  train = data[:-cutpoint]
  test = data[-cutpoint:]

  # --- Train, predict and forecast ---
  if tec == "lstm":
    res = lstm(train, test.size, shouldShowPlot)
  elif tec == "sarima":
    res = sarima(train, test.size, shouldShowPlot)
  elif tec == "sarimax":
    res = sarimax(train, test.size, shouldShowPlot)
  else:
    res = mlp(train, test.size, shouldShowPlot)

  #--- Recostruction ---
  exptrain = np.exp(res.train) # unlog
  expfore = np.exp(res.forecast)
  forecast_ci = res.forecast_ci
  if (forecast_ci != None):
    forecast_ci = ForecastCi(forecast_ci.index, np.exp(forecast_ci.max), np.exp(forecast_ci.min))
  res = ForecastResult(exptrain, expfore, forecast_ci, df)

  #--- Plot ---
  plot(res, shouldShowPlot)

  rawTrain = aValues[:-cutpoint]
  # --- Calculate revenue and risk ---
  actualValue = rawTrain[rawTrain.size - 1]
  lastMonthValues = res.forecast[-25:]
  forecastAvgValue = np.mean(lastMonthValues)
  accuracy_mape = np.mean(np.abs(res.forecast - rawTrain[len(-res.forecast):])/np.abs(rawTrain[len(-res.forecast):])) # MAPE

  # --- Value at Risk, historical simulation on the last 12 months ---
  pctChanges = pd.Series(rawTrain[-12*25-1:]).pct_change().dropna()
  pctChanges.sort_values(inplace=True, ascending=True)
  # VaR con confidenza 95%, lower perché altrimenti sceglie 16 invece di 15 valori
  # la funzione quantile divide l'array come richiesto e poi ritorna l'ultimo valore del primo quantile
  accuracy_var = pctChanges.quantile(0.05, interpolation='lower')

  print('LOG Last train value', actualValue)
  print('LOG Last forecasted month (avg)', forecastAvgValue)
  print('REVENUE', forecastAvgValue - actualValue)
  print('REVENUE_PERC', (forecastAvgValue-actualValue)/actualValue)
  print('MAPE', accuracy_mape)
  print('VAR', accuracy_var)
