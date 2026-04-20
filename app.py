import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="RouteWise - AI Transport Optimization System", layout="wide"
)

CSV_PATH = "smart_bus_data.csv"
BUS_CAPACITY = 60


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleaning
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "hour" in df.columns:
        if df["hour"].dtype == object:
            df["hour"] = df["hour"].astype(str).str[:2].astype(int)

    numeric_cols = [
        "lat",
        "lon",
        "passengers",
        "speed",
        "delay",
        "buses",
        "wait_time",
        "new_wait",
        "load",
        "new_load",
        "seg_dist",
        "cum_dist",
        "travel_time",
        "cum_travel_time",
        "efficiency",
        "is_peak",
        "overcrowded",
        "new_overcrowded",
        "stop_seq",
        "growth",
        "rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rebuild useful fields if missing
    if "load" not in df.columns and {"passengers", "buses"}.issubset(df.columns):
        df["load"] = df["passengers"] / BUS_CAPACITY
    if "new_load" not in df.columns and {"passengers", "buses"}.issubset(df.columns):
        df["new_load"] = df["passengers"] / (BUS_CAPACITY * df["buses"].clip(lower=1))
    if "overcrowded" not in df.columns and "load" in df.columns:
        df["overcrowded"] = (df["load"] > 1).astype(int)
    if "alert" not in df.columns and "load" in df.columns:
        df["alert"] = np.select(
            [df["load"] > 1, df["load"] > 0.8],
            ["Bus Full", "Almost Full"],
            default="Seats Available",
        )
    if "action" not in df.columns and {"load", "traffic", "delay"}.issubset(df.columns):
        df["action"] = np.select(
            [df["load"] > 1, df["traffic"].eq("Very High"), df["delay"] > 15],
            ["Add Bus", "Reroute", "Increase Frequency"],
            default="Normal",
        )
    if "day" not in df.columns and "date" in df.columns:
        df["day"] = df["date"].dt.day
    if "month" not in df.columns and "date" in df.columns:
        df["month"] = df["date"].dt.month
    if "weekday" not in df.columns and "date" in df.columns:
        df["weekday"] = df["date"].dt.day_name()

    return df.dropna(
        subset=[
            c for c in ["route", "stop", "lat", "lon", "passengers"] if c in df.columns
        ]
    )


@st.cache_resource
def train_models(df: pd.DataFrame):
    feature_cols = [
        c
        for c in [
            "hour",
            "route",
            "stop",
            "is_peak",
            "traffic",
            "lat",
            "lon",
            "stop_seq",
            "speed",
            "delay",
            "buses",
            "seg_dist",
            "cum_dist",
        ]
        if c in df.columns
    ]

    X = df[feature_cols].copy()
    y_reg = pd.to_numeric(df["passengers"], errors="coerce").fillna(0)

    y_cls = None
    if "action" in df.columns:
        y_cls = df["action"].astype(str).fillna("Normal")

    cat_cols = [
        c
        for c in X.columns
        if pd.api.types.is_object_dtype(X[c])
        or pd.api.types.is_string_dtype(X[c])
        or pd.api.types.is_categorical_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in cat_cols:
        X[c] = X[c].astype(str)

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg_model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42)),
        ]
    )
    reg_model.fit(X_train, y_train)

    reg_pred = reg_model.predict(X_test)
    reg_metrics = {
        "mae": float(mean_absolute_error(y_test, reg_pred)),
        "r2": float(r2_score(y_test, reg_pred)),
        "feature_cols": feature_cols,
    }

    cls_model, cls_metrics = None, None
    if y_cls is not None:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_cls, test_size=0.2, random_state=42
        )

        cls_model = Pipeline(
            [
                ("prep", preprocessor),
                ("model", RandomForestClassifier(n_estimators=220, random_state=42)),
            ]
        )
        cls_model.fit(Xc_train, yc_train)

        cls_pred = cls_model.predict(Xc_test)
        cls_metrics = {"accuracy": float(accuracy_score(yc_test, cls_pred))}

    return reg_model, reg_metrics, cls_model, cls_metrics


def status_from_load(load: float) -> str:
    if load > 1:
        return "Overcrowded 🚨"
    if load > 0.8:
        return "Moderate ⚠️"
    return "Normal ✅"


def apply_filters(df: pd.DataFrame, route, date_value, hour_range, traffic_values):
    filtered = df.copy()
    if route != "All":
        filtered = filtered[filtered["route"] == route]
    if date_value is not None and "date" in filtered.columns:
        filtered = filtered[filtered["date"].dt.date == date_value]
    if "hour" in filtered.columns:
        filtered = filtered[
            (filtered["hour"] >= hour_range[0]) & (filtered["hour"] <= hour_range[1])
        ]
    if traffic_values and "traffic" in filtered.columns and "All" not in traffic_values:
        filtered = filtered[filtered["traffic"].isin(traffic_values)]
    return filtered


def make_prediction_input(df: pd.DataFrame, route, stop, hour, traffic, is_peak):
    sample = df[(df["route"] == route) & (df["stop"] == stop)].sort_values("date")
    if sample.empty:
        sample = df[df["route"] == route].head(1)
    row = sample.iloc[0].copy()
    row["hour"] = hour
    row["traffic"] = traffic
    row["is_peak"] = is_peak
    return pd.DataFrame([row])


def simulate_extra_buses(predicted_passengers: float, current_buses: int):
    extra = st.slider("Add Extra Buses", 0, 5, 1)
    total_buses = max(current_buses + extra, 1)
    old_load = predicted_passengers / (BUS_CAPACITY * max(current_buses, 1))
    new_load = predicted_passengers / (BUS_CAPACITY * total_buses)
    base_wait = 10 if predicted_passengers > 60 else 5
    new_wait = round(base_wait / total_buses, 1)
    return extra, total_buses, old_load, new_load, new_wait


def main():
    st.title("🚍 RouteWise - AI Transport Optimization System")
    st.caption(
        "Real route dataset + ML demand prediction + smart operational recommendations"
    )

    try:
        df = load_data(CSV_PATH)
    except FileNotFoundError:
        st.error(
            f"CSV file not found: {CSV_PATH}. Save your dataset as smart_bus_data.csv in the same folder."
        )
        st.stop()

    reg_model, reg_metrics, cls_model, cls_metrics = train_models(df)

    routes = ["All"] + sorted(df["route"].dropna().unique().tolist())
    dates = (
        sorted(df["date"].dropna().dt.date.unique().tolist())
        if "date" in df.columns
        else []
    )
    traffic_options = (
        ["All"] + sorted(df["traffic"].dropna().unique().tolist())
        if "traffic" in df.columns
        else ["All"]
    )

    st.sidebar.header("🔍 Controls")
    selected_route = st.sidebar.selectbox(
        "Select Route", routes, index=1 if len(routes) > 1 else 0
    )
    selected_date = st.sidebar.selectbox("Date", dates, index=0 if dates else None)
    hour_range = st.sidebar.slider("Hour Range", 0, 23, (7, 20))
    selected_traffic = st.sidebar.multiselect(
        "Traffic", traffic_options, default=["All"]
    )

    filtered = apply_filters(
        df, selected_route, selected_date, hour_range, selected_traffic
    )
    if filtered.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    # Top KPIs
    total_routes = filtered["route"].nunique()
    total_stops = filtered["stop"].nunique()
    avg_passengers = int(round(filtered["passengers"].mean()))
    avg_delay = (
        round(filtered["delay"].mean(), 1) if "delay" in filtered.columns else 0.0
    )
    avg_speed = (
        round(filtered["speed"].mean(), 1) if "speed" in filtered.columns else 0.0
    )
    avg_load = round(filtered["load"].mean(), 2) if "load" in filtered.columns else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Routes", total_routes)
    k2.metric("Stops", total_stops)
    k3.metric("Avg Demand", avg_passengers)
    k4.metric("Avg Load", avg_load)
    k5.metric("Avg Delay", f"{avg_delay} min")
    k6.metric("Avg Speed", f"{avg_speed} km/h")

    left, right = st.columns([2.1, 1], gap="large")

    with left:
        st.subheader("🗺️ Route Map")
        map_df = filtered.sort_values(
            [c for c in ["route", "stop_seq", "hour"] if c in filtered.columns]
        )
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="passengers",
            size="passengers",
            hover_name="stop",
            hover_data={
                "route": True,
                "hour": True,
                "traffic": True if "traffic" in map_df.columns else False,
                "delay": True if "delay" in map_df.columns else False,
                "lat": False,
                "lon": False,
            },
            zoom=11,
            height=520,
            color_continuous_scale="Turbo",
        )
        fig_map.update_layout(
            mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Demand by Hour")
            demand_by_hour = filtered.groupby("hour", as_index=False)[
                "passengers"
            ].mean()
            fig_hour = px.line(demand_by_hour, x="hour", y="passengers", markers=True)
            fig_hour.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_hour, use_container_width=True)
        with c2:
            st.subheader("🔥 Demand Heatmap")
            heat = filtered.groupby(["route", "hour"], as_index=False)[
                "passengers"
            ].mean()
            fig_heat = px.density_heatmap(
                heat,
                x="hour",
                y="route",
                z="passengers",
                color_continuous_scale="YlOrRd",
            )
            fig_heat.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("📊 Route Comparison")
        route_avg = (
            filtered.groupby("route", as_index=False)["passengers"]
            .mean()
            .sort_values("passengers", ascending=False)
        )
        fig_bar = px.bar(
            route_avg,
            x="route",
            y="passengers",
            color="passengers",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.subheader("🧠 Smart Prediction")
        pred_route = selected_route if selected_route != "All" else df["route"].iloc[0]
        route_stops = sorted(df[df["route"] == pred_route]["stop"].unique().tolist())
        pred_stop = st.selectbox("Stop", route_stops)
        pred_hour = st.slider("Prediction Hour", 0, 23, 9)
        pred_traffic = st.selectbox(
            "Prediction Traffic",
            (
                sorted(df["traffic"].dropna().unique().tolist())
                if "traffic" in df.columns
                else ["Low", "Moderate", "High", "Very High"]
            ),
        )
        pred_is_peak = 1 if 7 <= pred_hour <= 10 or 17 <= pred_hour <= 20 else 0

        input_df = make_prediction_input(
            df, pred_route, pred_stop, pred_hour, pred_traffic, pred_is_peak
        )
        pred_passengers = float(
            reg_model.predict(input_df[reg_metrics["feature_cols"]])[0]
        )
        predicted_action = (
            cls_model.predict(input_df[reg_metrics["feature_cols"]])[0]
            if cls_model is not None
            else "Normal"
        )

        route_sample = df[(df["route"] == pred_route) & (df["stop"] == pred_stop)].head(
            1
        )
        current_buses = (
            int(route_sample["buses"].iloc[0]) if "buses" in route_sample.columns else 1
        )
        current_delay = (
            float(route_sample["delay"].iloc[0])
            if "delay" in route_sample.columns
            else 0.0
        )
        current_speed = (
            float(route_sample["speed"].iloc[0])
            if "speed" in route_sample.columns
            else 0.0
        )
        current_load = pred_passengers / (BUS_CAPACITY * max(current_buses, 1))

        st.metric("Predicted Demand", int(round(pred_passengers)))
        st.metric(
            "Predicted Load", round(current_load, 2), status_from_load(current_load)
        )
        st.metric("ML Recommendation", predicted_action)

        if current_load > 1:
            st.error("Add buses immediately. Current prediction shows overcrowding.")
        elif pred_traffic == "Very High":
            st.warning(
                "Traffic is very high. Consider rerouting or spacing departures."
            )
        else:
            st.success("Service looks stable for this route and time.")

        st.subheader("🔮 What-If Simulation")
        extra, total_buses, old_load, new_load, new_wait = simulate_extra_buses(
            pred_passengers, current_buses
        )
        st.write(f"Current buses: **{current_buses}**")
        st.write(f"After simulation: **{total_buses} buses**")
        st.write(f"Old load: **{old_load:.2f}**")
        st.write(f"New load: **{new_load:.2f}**")
        st.write(f"Estimated waiting time: **{new_wait} min**")
        if new_load < 0.8:
            st.success("Overcrowding solved.")
        else:
            st.warning("More optimization may still be needed.")

        st.subheader("📌 Operational Panel")
        st.info(f"Delay: {current_delay:.1f} min")
        st.info(f"Speed: {current_speed:.1f} km/h")
        if "alt_route" in route_sample.columns:
            st.info(f"Next best route: {route_sample['alt_route'].iloc[0]}")
        if "alert" in route_sample.columns:
            st.info(f"Passenger alert: {route_sample['alert'].iloc[0]}")

    st.subheader("📂 Filtered Data")
    st.dataframe(filtered, use_container_width=True, height=280)
    st.download_button(
        "Download Filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_smart_bus_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
