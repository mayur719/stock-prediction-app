# ========= Streamlit App =========
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import ta
from statsmodels.tsa.arima.model import ARIMA

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False


# ========= Functions =========
def feature_engineering(df):
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_high'] = ma20 + 2 * std20
    df['BB_low'] = ma20 - 2 * std20

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Stochastic'] = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close']).stoch()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        df['Close'], df['Volume']).on_balance_volume()
    df['ATR14'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['ROC12'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()

    df.dropna(inplace=True)
    return df


def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1)


def recursive_forecast(model, scaler, X, n_steps):
    future_preds, X_future = [], X.copy()
    for _ in range(n_steps):
        latest_scaled = scaler.transform(X_future.iloc[-1:].values)
        pred = model.predict(latest_scaled)[0]
        future_preds.append(pred)

        last_close = X_future['MA5'].iloc[-1]
        change = 0.01 if pred == 1 else -0.01
        new_close = last_close * (1 + change)
        new_row = X_future.iloc[-1].to_dict()
        new_row.update({'MA5': new_close, 'MA10': new_close,
                        'Momentum': new_close - last_close})
        X_future = pd.concat([X_future, pd.DataFrame([new_row])],
                             ignore_index=True)
    return np.array(future_preds)


# ========= Streamlit UI =========
st.set_page_config(page_title="Interactive Stock Prediction", layout="wide")
st.title("📈 Stock Direction Prediction")
st.markdown(
    """
    Predict stock **UP/DOWN** direction using multiple ML models and ARIMA forecasts.  
    Use the sidebar to adjust **model** and **forecast horizon**.
    """
)

# Sidebar
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (Open, High, Low, Close, Volume):", type="csv"
)
horizon = st.sidebar.slider("Select Forecast Horizon (days)", 1, 30, 5)

# Main content
if uploaded_file:
    # --- Data ---
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')

    with st.expander("📂 Raw Data Preview", expanded=False):
        st.dataframe(df.head())

    # --- Feature Engineering ---
    df = feature_engineering(df)
    features = [
        'MA5', 'MA10', 'Momentum', 'BB_high', 'BB_low', 'RSI', 'MACD',
        'Stochastic', 'OBV', 'ATR14', 'ROC12', 'Volume'
    ]
    X, y = df[features], df['Direction']

    # Split, scale, balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    scaler = StandardScaler()
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(
        scaler.fit_transform(X_train), y_train
    )
    X_test_scaled = scaler.transform(X_test)

    # --- Models ---
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "SVM_RBF": SVC(kernel='rbf', C=1.0, gamma='scale',
                       class_weight='balanced', probability=True, random_state=42),
        "KNN_15": KNeighborsClassifier(n_neighbors=15),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=400, learning_rate=0.05, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric='logloss', random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=700, num_leaves=31, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1)
    }
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=700, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, loss_function='Logloss', random_state=42, verbose=False
        )

    # Train & Evaluate
    results = []
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_scaled)
        results.append(evaluate(y_test, y_pred, name))

    # Stacking Ensemble
    stack_estimators = [(m, models[m]) for m in
                        ["RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM"] if m in models]
    stack_model = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
        stack_method='predict_proba', n_jobs=-1
    )
    stack_model.fit(X_train_bal, y_train_bal)
    results.append(evaluate(y_test, stack_model.predict(X_test_scaled), "STACK_Ensemble"))

    leaderboard = pd.DataFrame(results).sort_values(
        by=["f1", "accuracy"], ascending=False
    ).reset_index(drop=True)

    # Sidebar update after leaderboard is ready
    selected_model_name = st.sidebar.selectbox(
        "Select Model for Forecast", leaderboard['model'], index=0
    )

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["🔮 ML Forecast", "📈 ARIMA Forecast", "🏆 Leaderboard"])

    # --- ML Forecast ---
    with tab1:
        best_model = models.get(selected_model_name, stack_model)
        ml_preds = recursive_forecast(best_model, scaler, X, horizon)

        up_prob = ml_preds.mean()
        st.metric(label=f"{horizon}-day UP Probability", value=f"{up_prob:.2f}")

        # Prediction Info
        st.subheader("📊 Prediction Summary")
        up_days = np.sum(ml_preds == 1)
        down_days = np.sum(ml_preds == 0)
        st.write(f"Out of **{horizon} days** forecasted:")
        st.write(f"- 📈 **{up_days} days UP**")
        st.write(f"- 📉 **{down_days} days DOWN**")

        # Bar chart
        st.bar_chart(pd.Series(ml_preds).value_counts().rename({0: "DOWN", 1: "UP"}))

        # Timeline
        forecast_index = pd.date_range(
            start=X.index[-1] + pd.Timedelta(days=1),
            periods=horizon, freq="B"
        )
        forecast_df = pd.DataFrame({"Date": forecast_index, "Direction": ml_preds})
        forecast_df["Direction"] = forecast_df["Direction"].map({1: "UP", 0: "DOWN"})

        st.subheader("📅 Forecast Timeline")
        st.dataframe(forecast_df, use_container_width=True)

        # Narrative
        if up_prob > 0.6:
            st.success("The model indicates a **strong bullish trend** ahead.")
        elif up_prob < 0.4:
            st.error("The model suggests a **bearish outlook** ahead.")
        else:
            st.warning("The outlook is **mixed/uncertain**. Market may stay sideways.")

        # Combined chart: Actual Close + Forecast Directions
        st.subheader("📈 Combined Forecast Chart")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'][-50:], label="Actual Close", color="black")
        ax.scatter(forecast_index, df['Close'].iloc[-1] *
                   (1 + 0.01 * np.cumsum(np.where(ml_preds == 1, 1, -1))),
                   c=np.where(ml_preds == 1, "green", "red"),
                   label="Forecasted Direction", marker="o")
        ax.legend()
        st.pyplot(fig)

    # --- ARIMA Forecast ---
    # --- ARIMA Forecast ---
    with tab2:
        close_series = df['Close']
        forecast_index = pd.date_range(
            start=close_series.index[-1] + pd.Timedelta(days=1),
            periods=horizon, freq='B'
        )
        arima_model = ARIMA(close_series, order=(5, 1, 0)).fit()
        arima_forecast = arima_model.forecast(steps=horizon)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(close_series[-50:], label="Actual Close", color="black")
        ax.plot(forecast_index, arima_forecast, label="ARIMA Forecast", color="red")
        ax.set_title(f"Next {horizon}-Day ARIMA Forecast")
        ax.legend()
        st.pyplot(fig)

        # Show predicted values
        st.subheader("📅 ARIMA Forecasted Values")
        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            "Predicted_Close": arima_forecast
        })
        st.dataframe(forecast_df, use_container_width=True)

    # --- Leaderboard ---
    with tab3:
        st.dataframe(leaderboard, use_container_width=True)
