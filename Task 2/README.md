# Time Series Modeling
This task was to model an LSTM encoder-decoder and a SARIMAX model to use on Time series data.

## Dataset
The dataset used is the NASDAQ stock market data for 1month, 6months, 1 year, 5 years. This was done to show the quality of the LSTM model on different data sizes.

## LSTM encoder-decoder
LSTM stands for Long Short Term Memory and it is a type of recurrent neural network (RNN) architecture designed to model temporal sequences and their long-range dependencies more accurately than traditional RNNs. Traditional RNNs also have the exploding / vanishing gradients problem which makes them hard to train. LSTMs help combat this issue and are superior to RNNs when it comes to working with time series data. LSTM networks address the vanishing gradient problem common in traditional RNNs by incorporating memory cells and a gating mechanism.

The LSTM Encoder-Decoder architecture with attention is an advanced model designed to handle sequences of varying lengths, particularly useful in applications like machine translation, text summarization, and conversational modeling. This architecture enhances the basic Encoder-Decoder model by integrating an attention mechanism, which helps the model focus on relevant parts of the input sequence when generating each part of the output sequence.

The attention mechanism allows the decoder to selectively focus on different parts of the input sequence during the decoding process, rather than relying on a single fixed context vector. This improves the model's ability to handle long sequences and capture dependencies more effectively.

The code and implementation of the LSTM encoder decoder can be found in ```LSTM_encoder_decoder.ipynb```. 

Notice the effects of different data sizes:

1 month:
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%202/plots/Attention_LSTM_NASDAQ100_1month.png)

6 months:
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%202/plots/Attention_LSTM_NASDAQ100_6months.png)

1 year:
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%202/plots/Attention_LSTM_NASDAQ100_1year.png)

5 years:
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%202/plots/Attention_LSTM_NASDAQ100_5years.png)

## SARIMAX Model
SARIMAX stands for Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors. It is an advanced form of the ARIMA (AutoRegressive Integrated Moving Average) model, incorporating seasonal effects and the influence of external variables. This model is commonly used in time-series forecasting, especially when the data exhibits seasonality and is influenced by external factors.

The code can be found in ```SARIMAX.ipynb```
Grid Search was done to find the best parameters

Here are some graphs of the data and its forecasting

Data:
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%202/plots/SARIMAX_train_data.png)

Forecasting:
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%202/plots/SARIMAX_forecast.png)

While the LSTM model can take in a lot of features, SARIMAX is limited to only one feature. SARIMAX is only good for making forcasting for a short time frame while LSTM is good for longer time frames