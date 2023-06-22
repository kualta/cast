Demand forecasting for crypto using Facebook's `prophet`, Binance API with real time Web UI using `bokeh`

### Usage
1. Clone the repo
```
git clone git@github.com:kualta/cast
```
```
cd cast
```
2. Install the dependencies 
```
python3 -m venv venv
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```
3. Start the `bokeh` server
```
bokeh serve --show .
```
This will open a new window in your browser with the plot, start adjusting the parameters to refit the model

![image](https://github.com/kualta/cast/assets/72769566/66c111b7-531d-4a75-a8d6-12a3c8d89db0)
