import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import time

st.set_page_config(
    page_title="Weyland Yutani Dashboard",
    layout="wide"
)

st.title("Weyland Yutani Corporation")
st.subheader("Weyland Yutani Mining Dashboard")

SHEET_ID = "1KjilJ9werVWdCoP50YYVHEaPIqXXEZicax8pV4NHGcc"

@st.cache_data(ttl=30)
def load_data():
    GID = "479371700"

    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

    response = requests.get(
    url,
    params={"refresh": int(time.time())},
    headers={"Cache-Control": "no-cache"},
    timeout=20)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = df.columns.astype(str).str.strip()

    if "Date" not in df.columns:
        raise ValueError(f"Date column not found: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Google Sheets connection failed: {e}")
    st.stop()

mines = [
    col for col in df.columns
    if col not in ["Date", "Total Output"]
    and df[col].notna().any()
]

st.sidebar.header("Settings")

selected_mine = st.sidebar.selectbox(
    "Select Mine", ["All Mines"] + mines
)

chart_type = st.sidebar.selectbox(
    "Chart Type", ["Line", "Bar", "Stacked"]
)

trend_degree = st.sidebar.selectbox(
    "Trendline Degree", [1, 2, 3, 4]
)


st.sidebar.header("Anomaly Detection")

method = st.sidebar.selectbox(
    "Detection Method",
    ["IQR", "Z-score", "Moving Average", "Grubbs"]
)

if method == "IQR":
    iqr_factor = st.sidebar.slider("IQR Factor", 0.5, 5.0, 1.5, 0.1)

elif method == "Z-score":
    z_threshold = st.sidebar.slider("Z-score Threshold", 1.0, 5.0, 3.0, 0.1)

elif method == "Moving Average":
    window = st.sidebar.slider("Window Size", 3, 30, 7)
    ma_threshold = st.sidebar.slider("Deviation Threshold", 0.05, 1.0, 0.3, 0.05)

elif method == "Grubbs":
    alpha = st.sidebar.selectbox("Significance Level", [0.01, 0.05, 0.1], index=1)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.success("Google Sheets connected")

st.write(f"Rows loaded: {len(df)}")
st.write(f"Active mines: {len(mines)}")

st.subheader("Mining Data")
st.dataframe(df, use_container_width=True)


df["Total Output"] = df[mines].sum(axis=1, min_count=1)

stats_data = []

for mine in mines + ["Total Output"]:
    values = df[mine].dropna()

    if len(values) == 0:
        continue

    mean = values.mean()
    std = values.std()
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    stats_data.append({
        "Mine": mine,
        "Mean": round(mean, 2),
        "Std": round(std, 2),
        "Median": round(median, 2),
        "IQR": round(iqr, 2)
    })

stats_df = pd.DataFrame(stats_data)

st.subheader("Mining Statistics")
st.dataframe(stats_df, use_container_width=True)

total_output = df["Total Output"].sum()
average_output = df["Total Output"].mean()
max_output = df["Total Output"].max()
min_output = df["Total Output"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Production", f"{total_output:,.0f}")
col2.metric("Average Daily Output", f"{average_output:,.0f}")
col3.metric("Maximum Daily Output", f"{max_output:,.0f}")
col4.metric("Minimum Daily Output", f"{min_output:,.0f}")


def detect_iqr(values, factor):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    return (values < lower) | (values > upper)


def detect_zscore(values, threshold):
    std = values.std()

    if std == 0 or pd.isna(std):
        return pd.Series(False, index=values.index)

    z = (values - values.mean()).abs() / std

    return z > threshold


def detect_moving_average(values, window, threshold):
    avg = values.rolling(window, min_periods=window).mean().shift(1)
    deviation = (values - avg).abs() / avg.abs().replace(0, np.nan)

    return (deviation > threshold).fillna(False)


def detect_grubbs(values, alpha):
    result = pd.Series(False, index=values.index)

    remaining = values.dropna().copy()

    while len(remaining) >= 3:
        n = len(remaining)
        std = remaining.std()

        if std == 0 or pd.isna(std):
            break

        deviations = (remaining - remaining.mean()).abs()
        index = deviations.idxmax()
        G = deviations.max() / std

        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        critical = ((n - 1) / np.sqrt(n)) * np.sqrt(
            t ** 2 / (n - 2 + t ** 2)
        )

        if G <= critical:
            break

        result.loc[index] = True
        remaining = remaining.drop(index)

    return result


if selected_mine == "All Mines":
    column = "Total Output"
else:
    column = selected_mine

values = df[column]

if method == "IQR":
    anomalies = detect_iqr(values, iqr_factor)

elif method == "Z-score":
    anomalies = detect_zscore(values, z_threshold)

elif method == "Moving Average":
    anomalies = detect_moving_average(values, window, ma_threshold)

else:
    anomalies = detect_grubbs(values, alpha)

df["Anomaly"] = anomalies.fillna(False)

anomaly_df = df[df["Anomaly"]].copy()

st.subheader("Anomaly Detection Results")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Mine", column)
col2.metric("Detection Method", method)
col3.metric("Anomalies Found", len(anomaly_df))

if len(anomaly_df) > 0:
    st.dataframe(
        anomaly_df[["Date", column]],
        use_container_width=True
    )
else:
    st.success("No anomalies detected")


st.subheader("Mining Production Charts")

if selected_mine == "All Mines":
    chart_mines = mines
else:
    chart_mines = [selected_mine]

fig = go.Figure()

for mine in chart_mines:
    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[mine],
            mode="lines",
            name=mine
        ))

    elif chart_type == "Bar":
        fig.add_trace(go.Bar(
            x=df["Date"],
            y=df[mine],
            name=mine
        ))

    else:
        fig.add_trace(go.Bar(
            x=df["Date"],
            y=df[mine],
            name=mine
        ))

if chart_type == "Stacked":
    fig.update_layout(barmode="stack")
elif chart_type == "Bar":
    fig.update_layout(barmode="group")

if selected_mine == "All Mines":
    st.caption("Anomalies and trendline use Total Output")
else:
    st.caption(f"Anomalies and trendline use {selected_mine}")

show_anomalies = st.checkbox("Show Anomalies", value=True)
show_trendline = st.checkbox("Show Polynomial Trendline", value=True)

if show_anomalies:
    fig.add_trace(go.Scatter(
        x=anomaly_df["Date"],
        y=anomaly_df[column],
        mode="markers",
        name="Anomalies",
        marker=dict(
            color="red",
            size=12,
            symbol="x"
        )
    ))

if show_trendline:
    x = np.arange(len(df))
    y = df[column].to_numpy(dtype=float)

    valid = np.isfinite(y)

    if valid.sum() > trend_degree:
        coefficients = np.polyfit(
            x[valid],
            y[valid],
            trend_degree
        )

        trend = np.polyval(coefficients, x)

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=trend,
            mode="lines",
            name=f"Polynomial Degree {trend_degree}",
            line=dict(
                color="orange",
                width=3,
                dash="dash"
            )
        ))

fig.update_layout(
    title="Daily Mining Production",
    xaxis_title="Date",
    yaxis_title="Production",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)


def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Weyland Yutani Corporation", styles["Title"]))
    elements.append(Paragraph("Mining Operations Analytics Report", styles["Heading2"]))
    elements.append(Spacer(1, 15))

    start = df["Date"].min().strftime("%Y-%m-%d")
    end = df["Date"].max().strftime("%Y-%m-%d")

    elements.append(Paragraph(f"Period: {start} to {end}", styles["Normal"]))
    elements.append(Paragraph(f"Active Mines: {len(mines)}", styles["Normal"]))
    elements.append(Paragraph(f"Total Production: {total_output:,.0f}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("1. Mining Statistics", styles["Heading2"]))

    table_data = [["Mine", "Mean", "Std", "Median", "IQR"]]

    for _, row in stats_df.iterrows():
        table_data.append([
            row["Mine"],
            f'{row["Mean"]:,.2f}',
            f'{row["Std"]:,.2f}',
            f'{row["Median"]:,.2f}',
            f'{row["IQR"]:,.2f}'
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    results = {}

    for mine in mines + ["Total Output"]:
        values = df[mine]

        mine_results = {
            "IQR": detect_iqr(
                values,
                iqr_factor if method == "IQR" else 1.5
            ),
            "Z-score": detect_zscore(
                values,
                z_threshold if method == "Z-score" else 3.0
            ),
            "Moving Average": detect_moving_average(
                values,
                window if method == "Moving Average" else 7,
                ma_threshold if method == "Moving Average" else 0.3
            ),
            "Grubbs": detect_grubbs(
                values,
                alpha if method == "Grubbs" else 0.05
            )
        }

        results[mine] = mine_results

    elements.append(Paragraph("2. Anomaly Detection Summary", styles["Heading2"]))

    summary = [["Mine", "IQR", "Z-score", "Moving Avg", "Grubbs"]]

    for mine, methods in results.items():
        summary.append([
            mine,
            str(methods["IQR"].sum()),
            str(methods["Z-score"].sum()),
            str(methods["Moving Average"].sum()),
            str(methods["Grubbs"].sum())
        ])

    table = Table(summary, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))

    elements.append(table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        "Default parameters: IQR factor 1.5, Z-score threshold 3.0, "
        "Moving Average window 7 and deviation 30%, Grubbs alpha 0.05. "
        "The currently selected method uses its sidebar settings.",
        styles["Normal"]
    ))

    elements.append(PageBreak())
    elements.append(Paragraph("3. Production Charts", styles["Heading2"]))

    for mine in mines + ["Total Output"]:
        fig_pdf, ax = plt.subplots(figsize=(8, 3.3))

        ax.plot(df["Date"], df[mine], label=mine)

        marks = results[mine][method].fillna(False)

        ax.scatter(
            df.loc[marks, "Date"],
            df.loc[marks, mine],
            color="red",
            label="Anomalies",
            zorder=3
        )

        x = np.arange(len(df))
        y = df[mine].to_numpy(dtype=float)
        valid = np.isfinite(y)

        if valid.sum() > trend_degree:
            coefficients = np.polyfit(x[valid], y[valid], trend_degree)
            trend = np.polyval(coefficients, x)

            ax.plot(
                df["Date"],
                trend,
                color="orange",
                linestyle="--",
                label=f"Polynomial {trend_degree}"
            )

        ax.set_title(mine)
        ax.set_xlabel("Date")
        ax.set_ylabel("Production")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        fig_pdf.autofmt_xdate()
        fig_pdf.tight_layout()

        img_buffer = BytesIO()
        fig_pdf.savefig(img_buffer, format="png", dpi=120)
        plt.close(fig_pdf)

        img_buffer.seek(0)

        elements.append(Image(img_buffer, width=510, height=210))
        elements.append(Spacer(1, 10))

    elements.append(PageBreak())
    elements.append(Paragraph("4. Detailed Anomaly Results", styles["Heading2"]))

    explanations = {
        "IQR": "The value is outside the interquartile range limits.",
        "Z-score": "The value is unusually far from the mean.",
        "Moving Average": "The value differs significantly from previous daily production.",
        "Grubbs": "The value was identified as an outlier by Grubbs' statistical test."
    }

    found = 0

    for mine, methods in results.items():
        for name, marks in methods.items():
            anomaly_rows = df[marks.fillna(False)]

            for _, row in anomaly_rows.iterrows():
                found += 1

                date = row["Date"].strftime("%Y-%m-%d")
                value = row[mine]

                elements.append(Paragraph(
                    f"<b>{mine} | {name} | {date}</b>",
                    styles["Normal"]
                ))

                elements.append(Paragraph(
                    f"Production: {value:,.2f}. {explanations[name]}",
                    styles["Normal"]
                ))

                elements.append(Spacer(1, 7))

    if found == 0:
        elements.append(Paragraph(
            "No anomalies detected with the selected parameters.",
            styles["Normal"]
        ))

    doc.build(elements)
    buffer.seek(0)

    return buffer.getvalue()


st.subheader("Download PDF Report")

pdf_data = generate_pdf()

st.download_button(
    label="Download PDF Report",
    data=pdf_data,
    file_name="Weyland_Yutani_Report.pdf",
    mime="application/pdf"
)
