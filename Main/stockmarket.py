import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Đọc dữ liệu từ file csv
df = pd.read_csv(r'D:\Users\ADMIN\Downloads\TITAN.csv')


""" CLEANING """
# Loại bỏ các giá trị trùng lặp
df.drop_duplicates(inplace=True)

# Xử lý các giá trị bị thiếu
df.dropna(inplace=True)

# Định dạng dữ liệu theo cách phù hợp để phân tích
df['Date'] = pd.to_datetime(df['Date'])
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(int)

# Chia dữ liệu thành các tập huấn luyện và kiểm tra
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


""" DATA MINING """
# Vẽ biểu đồ giá cổ phiếu theo thời gian bằng Seaborn(1.png)
sns.lineplot(x='Date', y='Close', data=df)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Historical Stock Prices')
plt.show()

# Tính toán đường trung bình động trong 30 ngày
rolling_avg = df['Close'].rolling(window=30).mean()

# Vẽ biểu đồ giá cổ phiếu và đường trung bình động bằng Seaborn(2.png)
df['30-day Moving Average'] = rolling_avg
sns.lineplot(x='Date', y='Close', data=df, label='Close Price')
sns.lineplot(x='Date', y='30-day Moving Average', data=df, label='30-day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Historical Stock Prices')
plt.legend()
plt.show()

# Vẽ biểu đồ khối lượng giao dịch bằng Seaborn(3.png)
sns.barplot(x='Date', y='Volume', data=df)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Historical Trading Volume')
plt.show()


# Tính toán đường trung bình động trong 30 ngày
df['30-day Moving Average'] = df['Close'].rolling(window=30).mean()

# Tính toán dải Bollinger
rolling_std = df['Close'].rolling(window=30).std()
df['Upper Band'] = df['30-day Moving Average'] + 2 * rolling_std
df['Lower Band'] = df['30-day Moving Average'] - 2 * rolling_std

# Tính toán chỉ số RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Vẽ biểu đồ giá cổ phiếu và các chỉ số kỹ thuật tương ứng(4.png)
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'], label='Close Price')
ax.plot(df['Date'], df['30-day Moving Average'], label='30-day Moving Average')
ax.plot(df['Date'], df['Upper Band'], label='Upper Band')
ax.plot(df['Date'], df['Lower Band'], label='Lower Band')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Historical Stock Prices and Technical Indicators')
ax.legend()
ax2 = ax.twinx()
ax2.plot(df['Date'], df['RSI'], color='purple', label='RSI')
ax2.set_ylabel('RSI')
ax2.legend()
plt.show()


""" MODELING """
# Chuẩn bị dữ liệu
df = pd.read_csv(r'D:\Users\ADMIN\Downloads\TITAN.csv')
df['Close'] = df['Close'].astype(float) 
df.index = pd.to_datetime(df['Date'])
df = df.resample('D').ffill()
X = df['Close']
train_size = int(len(X)*0.9)
train, test = X[1:train_size], X[train_size:len(X)]

# Xây dựng mô hình ARIMA
model = ARIMA(train, order=(1, 2, 1))
model_fit = model.fit()

# Đánh giá hisuất của mô hình
history = [x for x in train]
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(1, 2, 1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)


mse = mean_squared_error(test, predictions)
print('Mean Squared Error:', mse)

# Trực quan hóa kết quả
plt.plot(test.index, test.values, label='Giá thực tế')
plt.plot(test.index, predictions, label='Giá dự đoán')
plt.legend()
plt.title('Mô hình ARIMA - MSE: {:.4f}'.format(mse))
plt.xlabel('giá')
plt.ylabel('Nước')
plt.show()
