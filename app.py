import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# LOAD DATA (CHANGE FILE NAME HERE)
# -----------------------------
df = pd.read_csv("data.csv")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M"
)



# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
SAFE_LIMIT = 8

df["Above_Safe"] = df["Temperature"] > SAFE_LIMIT
df["Temp_Change"] = df["Temperature"].diff().fillna(0)

time_outside = df["Above_Safe"].sum() * 5  # assuming 5-minute intervals
max_temp = df["Temperature"].max()
avg_temp = df["Temperature"].mean()

# -----------------------------
# LABEL LOGIC
# -----------------------------
def label_risk(row):
    if row["Temperature"] <= 8:
        return 0  # Safe
    elif row["Temperature"] <= 10:
        return 1  # Warning
    else:
        return 2  # Compromised

df["Risk_Label"] = df.apply(label_risk, axis=1)

# -----------------------------
# CLASSIFICATION MODEL
# -----------------------------
X = df[["Temperature", "Temp_Change"]]
y = df["Risk_Label"]

clf = DecisionTreeClassifier()
clf.fit(X, y)

df["Predicted_Risk"] = clf.predict(X)

# -----------------------------
# REGRESSION (Shelf Life Impact)
# -----------------------------
df["Exposure_Time"] = df["Above_Safe"].cumsum()
reg = LinearRegression()
reg.fit(df[["Exposure_Time"]], df["Temperature"])

# -----------------------------
# TIME SERIES FORECASTING
# -----------------------------
df["Time_Index"] = np.arange(len(df))
ts_model = LinearRegression()
ts_model.fit(df[["Time_Index"]], df["Temperature"])

future_index = np.array([[len(df) + i] for i in range(10)])
future_temp = ts_model.predict(future_index)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧊 Cold-Chain Risk Monitoring System")

st.subheader("📊 Temperature Overview")
st.line_chart(df.set_index("Timestamp")["Temperature"])

st.subheader("⚠️ Risk Summary")
st.write(f"**Average Temperature:** {avg_temp:.2f} °C")
st.write(f"**Maximum Temperature:** {max_temp:.2f} °C")
st.write(f"**Time Outside Safe Range:** {time_outside} minutes")

risk_map = {0: "Safe", 1: "Warning", 2: "Compromised"}
final_risk = risk_map[df["Predicted_Risk"].iloc[-1]]

st.subheader("🚨 Final Shipment Status")
st.success(final_risk)

st.subheader("🔮 Future Temperature Prediction")
fig, ax = plt.subplots()
ax.plot(df["Temperature"], label="Actual")
ax.plot(range(len(df), len(df) + 10), future_temp, label="Predicted", linestyle="--")
ax.legend()
st.pyplot(fig)
