import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# -------------------------------------------------
# Monitoring points for vulnerable northern valleys
# Approximate valley points. Later we can upgrade
# these to full basin polygons.
# -------------------------------------------------
VALLEYS = [
    {"name": "Hunza_Gojal", "lat": 36.45, "lon": 74.85},
    {"name": "Gulmit_Gulkin", "lat": 36.38, "lon": 74.86},
    {"name": "Ishkoman", "lat": 36.75, "lon": 73.85},
    {"name": "Yasin", "lat": 36.45, "lon": 73.32},
    {"name": "Shigar", "lat": 35.43, "lon": 75.73},
    {"name": "Reshun_Chitral", "lat": 36.15, "lon": 72.15},
    {"name": "Brep_Yarkhun", "lat": 36.37, "lon": 72.35},
    {"name": "Kumrat", "lat": 35.55, "lon": 72.23},
]

# -------------------------------------------------
# Prototype thresholds.
# These are NOT official warning thresholds.
# They are screening thresholds for research prototype.
# -------------------------------------------------
THRESHOLDS = {
    # High mountain air temperature thresholds
    "observed_heat_watch_c": 24,
    "observed_heat_warning_c": 30,
    "observed_heat_critical_c": 35,

    "forecast_heat_watch_c": 25,
    "forecast_heat_warning_c": 31,
    "forecast_heat_critical_c": 36,

    # Rainfall thresholds
    "observed_rain_watch_24h_mm": 10,
    "observed_rain_warning_24h_mm": 25,
    "observed_rain_critical_24h_mm": 50,

    "forecast_rain_watch_48h_mm": 20,
    "forecast_rain_warning_48h_mm": 45,
    "forecast_rain_critical_48h_mm": 80,

    # Snow/melt proxy thresholds
    "melt_temp_watch_c": 5,
    "melt_temp_warning_c": 10,
    "melt_temp_critical_c": 15,
}

RISK_POINTS = {
    "NORMAL": 0,
    "WATCH": 1,
    "WARNING": 2,
    "CRITICAL": 3,
    "UNKNOWN": 0,
}

WEIGHTS = {
    "ObservedHeatAgent": 1.2,
    "ObservedRainfallAgent": 1.8,
    "ForecastHeatAgent": 1.4,
    "ForecastRainfallAgent": 2.0,
    "SnowMeltForecastAgent": 1.7,
    "FloodRiskAgent": 2.5,
    "LandslideRiskAgent": 2.2,
}


def yyyymmdd(date_obj):
    return date_obj.strftime("%Y%m%d")


# -------------------------------------------------
# NASA POWER recent hourly data
# -------------------------------------------------
def fetch_nasa_power_hourly(lat, lon):
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=3)

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=T2M,PRECTOTCORR,RH2M,WS10M"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={yyyymmdd(start_date)}"
        f"&end={yyyymmdd(end_date)}"
        f"&format=JSON"
        f"&time-standard=UTC"
    )

    response = requests.get(url, timeout=40)
    response.raise_for_status()
    data = response.json()

    params = data["properties"]["parameter"]
    rows = []

    for ts in params["T2M"].keys():
        rows.append({
            "timestamp": ts,
            "temperature_c": params["T2M"].get(ts),
            "precip_mm": params["PRECTOTCORR"].get(ts),
            "humidity_percent": params["RH2M"].get(ts),
            "wind_speed_mps": params["WS10M"].get(ts),
        })

    df = pd.DataFrame(rows)

    for col in ["temperature_c", "precip_mm", "humidity_percent", "wind_speed_mps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -------------------------------------------------
# Open-Meteo forecast data
# -------------------------------------------------
def fetch_open_meteo_forecast(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,"
        f"snowfall,snow_depth,wind_speed_10m"
        f"&forecast_days=3"
        f"&timezone=UTC"
    )

    response = requests.get(url, timeout=40)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": hourly["time"],
        "forecast_temperature_c": hourly.get("temperature_2m", []),
        "forecast_humidity_percent": hourly.get("relative_humidity_2m", []),
        "forecast_precip_mm": hourly.get("precipitation", []),
        "forecast_snowfall_cm": hourly.get("snowfall", []),
        "forecast_snow_depth_m": hourly.get("snow_depth", []),
        "forecast_wind_speed_kmh": hourly.get("wind_speed_10m", []),
    })

    for col in [
        "forecast_temperature_c",
        "forecast_humidity_percent",
        "forecast_precip_mm",
        "forecast_snowfall_cm",
        "forecast_snow_depth_m",
        "forecast_wind_speed_kmh",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def classify_value(value, watch, warning, critical):
    if value is None or pd.isna(value):
        return "UNKNOWN"
    if value >= critical:
        return "CRITICAL"
    elif value >= warning:
        return "WARNING"
    elif value >= watch:
        return "WATCH"
    return "NORMAL"


# -------------------------------------------------
# Agents
# -------------------------------------------------
def observed_heat_agent(nasa_df):
    max_temp = round(float(nasa_df["temperature_c"].max()), 2)
    mean_temp = round(float(nasa_df["temperature_c"].mean()), 2)

    risk = classify_value(
        max_temp,
        THRESHOLDS["observed_heat_watch_c"],
        THRESHOLDS["observed_heat_warning_c"],
        THRESHOLDS["observed_heat_critical_c"],
    )

    return {
        "agent": "ObservedHeatAgent",
        "risk": risk,
        "max_temperature_c": max_temp,
        "mean_temperature_c": mean_temp,
        "summary": f"Recent maximum observed temperature: {max_temp} C",
    }


def observed_rainfall_agent(nasa_df):
    recent_24 = nasa_df.tail(24)
    rain_24h = round(float(recent_24["precip_mm"].sum()), 2)

    risk = classify_value(
        rain_24h,
        THRESHOLDS["observed_rain_watch_24h_mm"],
        THRESHOLDS["observed_rain_warning_24h_mm"],
        THRESHOLDS["observed_rain_critical_24h_mm"],
    )

    return {
        "agent": "ObservedRainfallAgent",
        "risk": risk,
        "observed_rainfall_24h_mm": rain_24h,
        "summary": f"Observed/near-real rainfall in last 24 hours: {rain_24h} mm",
    }


def forecast_heat_agent(openmeteo_df):
    next_48 = openmeteo_df.head(48)
    max_temp_48h = round(float(next_48["forecast_temperature_c"].max()), 2)

    risk = classify_value(
        max_temp_48h,
        THRESHOLDS["forecast_heat_watch_c"],
        THRESHOLDS["forecast_heat_warning_c"],
        THRESHOLDS["forecast_heat_critical_c"],
    )

    return {
        "agent": "ForecastHeatAgent",
        "risk": risk,
        "forecast_max_temperature_48h_c": max_temp_48h,
        "summary": f"Forecast maximum temperature in next 48h: {max_temp_48h} C",
    }


def forecast_rainfall_agent(openmeteo_df):
    next_48 = openmeteo_df.head(48)
    rain_48h = round(float(next_48["forecast_precip_mm"].sum()), 2)

    risk = classify_value(
        rain_48h,
        THRESHOLDS["forecast_rain_watch_48h_mm"],
        THRESHOLDS["forecast_rain_warning_48h_mm"],
        THRESHOLDS["forecast_rain_critical_48h_mm"],
    )

    return {
        "agent": "ForecastRainfallAgent",
        "risk": risk,
        "forecast_rainfall_48h_mm": rain_48h,
        "summary": f"Forecast precipitation in next 48h: {rain_48h} mm",
    }


def snow_melt_forecast_agent(openmeteo_df):
    next_48 = openmeteo_df.head(48)

    max_temp = round(float(next_48["forecast_temperature_c"].max()), 2)
    snow_depth_max = round(float(next_48["forecast_snow_depth_m"].max()), 3)
    snowfall_total = round(float(next_48["forecast_snowfall_cm"].sum()), 2)

    # Melt proxy: if temperature is above freezing and snow exists, risk rises.
    if snow_depth_max > 0:
        risk = classify_value(
            max_temp,
            THRESHOLDS["melt_temp_watch_c"],
            THRESHOLDS["melt_temp_warning_c"],
            THRESHOLDS["melt_temp_critical_c"],
        )
    else:
        # Even if Open-Meteo does not show snow depth at the point,
        # high mountain glacier zones around it can still melt in heat.
        if max_temp >= THRESHOLDS["forecast_heat_warning_c"]:
            risk = "WATCH"
        else:
            risk = "NORMAL"

    return {
        "agent": "SnowMeltForecastAgent",
        "risk": risk,
        "forecast_max_temperature_48h_c": max_temp,
        "forecast_snow_depth_max_m": snow_depth_max,
        "forecast_snowfall_total_cm": snowfall_total,
        "summary": (
            f"Melt proxy: max temp {max_temp} C, "
            f"snow depth {snow_depth_max} m, snowfall {snowfall_total} cm"
        ),
    }


def flood_risk_agent(observed_rain, forecast_rain, melt):
    observed_points = RISK_POINTS.get(observed_rain["risk"], 0)
    forecast_points = RISK_POINTS.get(forecast_rain["risk"], 0)
    melt_points = RISK_POINTS.get(melt["risk"], 0)

    combined_points = observed_points + forecast_points + melt_points

    if combined_points >= 7:
        risk = "CRITICAL"
    elif combined_points >= 5:
        risk = "WARNING"
    elif combined_points >= 2:
        risk = "WATCH"
    else:
        risk = "NORMAL"

    return {
        "agent": "FloodRiskAgent",
        "risk": risk,
        "combined_trigger_points": combined_points,
        "summary": "Flood risk from observed rainfall + forecast rainfall + snow/glacier melt proxy",
    }


def landslide_risk_agent(observed_rain, forecast_rain):
    observed_points = RISK_POINTS.get(observed_rain["risk"], 0)
    forecast_points = RISK_POINTS.get(forecast_rain["risk"], 0)

    combined_points = observed_points + forecast_points

    if combined_points >= 5:
        risk = "CRITICAL"
    elif combined_points >= 3:
        risk = "WARNING"
    elif combined_points >= 1:
        risk = "WATCH"
    else:
        risk = "NORMAL"

    return {
        "agent": "LandslideRiskAgent",
        "risk": risk,
        "combined_trigger_points": combined_points,
        "summary": "Landslide trigger proxy from recent and forecast rainfall",
    }


def risk_fusion_agent(agent_results):
    score = 0
    reasons = []

    for result in agent_results:
        agent = result["agent"]
        risk = result["risk"]

        points = RISK_POINTS.get(risk, 0)
        weight = WEIGHTS.get(agent, 1.0)
        weighted_score = points * weight

        score += weighted_score

        if risk != "NORMAL":
            reasons.append({
                "agent": agent,
                "risk": risk,
                "weighted_score": round(weighted_score, 2),
                "summary": result.get("summary", ""),
            })

    if score >= 16:
        level = "CRITICAL"
    elif score >= 10:
        level = "WARNING"
    elif score >= 4:
        level = "WATCH"
    else:
        level = "NORMAL"

    return {
        "risk_level": level,
        "score": round(score, 2),
        "reasons": reasons,
    }


def run_for_valley(valley):
    print(f"Running real-data agents for {valley['name']}")

    nasa_df = fetch_nasa_power_hourly(valley["lat"], valley["lon"])
    forecast_df = fetch_open_meteo_forecast(valley["lat"], valley["lon"])

    observed_heat = observed_heat_agent(nasa_df)
    observed_rain = observed_rainfall_agent(nasa_df)

    forecast_heat = forecast_heat_agent(forecast_df)
    forecast_rain = forecast_rainfall_agent(forecast_df)

    melt = snow_melt_forecast_agent(forecast_df)

    flood = flood_risk_agent(observed_rain, forecast_rain, melt)
    landslide = landslide_risk_agent(observed_rain, forecast_rain)

    agent_results = [
        observed_heat,
        observed_rain,
        forecast_heat,
        forecast_rain,
        melt,
        flood,
        landslide,
    ]

    combined = risk_fusion_agent(agent_results)

    return {
        "name": valley["name"],
        "lat": valley["lat"],
        "lon": valley["lon"],
        "agents": agent_results,
        "combined_risk": combined,
    }


def save_outputs(final_result):
    os.makedirs("data", exist_ok=True)

    with open("data/latest_result.json", "w") as f:
        json.dump(final_result, f, indent=2)

    history_path = "data/history.csv"

    row = {
        "timestamp": final_result["timestamp"],
        "risk_level": final_result["combined_risk"]["risk_level"],
        "score": final_result["combined_risk"]["score"],
        "valleys_monitored": len(final_result["valleys"]),
    }

    if os.path.exists(history_path):
        old = pd.read_csv(history_path)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])

    new.to_csv(history_path, index=False)

    valley_rows = []

    for valley in final_result["valleys"]:
        valley_rows.append({
            "timestamp": final_result["timestamp"],
            "valley": valley["name"],
            "lat": valley["lat"],
            "lon": valley["lon"],
            "risk_level": valley["combined_risk"]["risk_level"],
            "score": valley["combined_risk"]["score"],
        })

    pd.DataFrame(valley_rows).to_csv("data/latest_valleys.csv", index=False)


def main():
    valley_results = []

    for valley in VALLEYS:
        try:
            valley_result = run_for_valley(valley)
            valley_results.append(valley_result)
        except Exception as e:
            print(f"ERROR in {valley['name']}: {e}")
            valley_results.append({
                "name": valley["name"],
                "lat": valley["lat"],
                "lon": valley["lon"],
                "error": str(e),
                "agents": [],
                "combined_risk": {
                    "risk_level": "UNKNOWN",
                    "score": 0,
                    "reasons": [],
                },
            })

    # Overall fusion from valley scores
    valley_level_results = []

    for valley in valley_results:
        valley_level_results.append({
            "agent": f"ValleyRisk_{valley['name']}",
            "risk": valley["combined_risk"]["risk_level"],
            "summary": f"{valley['name']} score {valley['combined_risk']['score']}",
        })

        for reason in valley["combined_risk"].get("reasons", []):
            reason["valley"] = valley["name"]

    overall = risk_fusion_agent(valley_level_results)

    final_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "Northern Pakistan Glacial Flood and Landslide Watch",
        "data_sources": [
            "NASA POWER Hourly API",
            "Open-Meteo Forecast API",
        ],
        "combined_risk": overall,
        "valleys": valley_results,
    }

    save_outputs(final_result)

    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()