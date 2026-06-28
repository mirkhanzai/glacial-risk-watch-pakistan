import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# ---------------------------------------------
# Valley monitoring points
# You can add more locations later.
# lat/lon are approximate monitoring points.
# ---------------------------------------------
VALLEYS = [
    {
        "name": "Hunza_Gojal",
        "lat": 36.45,
        "lon": 74.85
    },
    {
        "name": "Gulmit_Gulkin",
        "lat": 36.38,
        "lon": 74.86
    },
    {
        "name": "Ishkoman",
        "lat": 36.75,
        "lon": 73.85
    },
    {
        "name": "Yasin",
        "lat": 36.45,
        "lon": 73.32
    },
    {
        "name": "Shigar",
        "lat": 35.43,
        "lon": 75.73
    },
    {
        "name": "Reshun_Chitral",
        "lat": 36.15,
        "lon": 72.15
    },
    {
        "name": "Brep_Yarkhun",
        "lat": 36.37,
        "lon": 72.35
    },
    {
        "name": "Kumrat",
        "lat": 35.55,
        "lon": 72.23
    }
]

# ---------------------------------------------
# Thresholds for prototype
# These are not official warning thresholds.
# You must calibrate with local data later.
# ---------------------------------------------
THRESHOLDS = {
    "heat_watch_c": 25,
    "heat_warning_c": 30,
    "heat_critical_c": 35,

    "rain_watch_mm_24h": 10,
    "rain_warning_mm_24h": 25,
    "rain_critical_mm_24h": 50
}

RISK_POINTS = {
    "NORMAL": 0,
    "WATCH": 1,
    "WARNING": 2,
    "CRITICAL": 3
}

WEIGHTS = {
    "HeatAgent": 1.4,
    "RainfallAgent": 1.8,
    "GlacierMeltProxyAgent": 1.5,
    "FloodProxyAgent": 2.2,
    "LandslideProxyAgent": 2.0
}


def yyyymmdd(date_obj):
    return date_obj.strftime("%Y%m%d")


def fetch_nasa_power_hourly(lat, lon):
    """
    Fetch recent hourly temperature and precipitation from NASA POWER.
    Parameters:
    T2M = temperature at 2 meters
    PRECTOTCORR = corrected precipitation
    RH2M = relative humidity at 2 meters
    WS10M = wind speed at 10 meters
    """

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=2)

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

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    params = data["properties"]["parameter"]

    rows = []

    for timestamp in params["T2M"].keys():
        rows.append({
            "timestamp": timestamp,
            "temperature_c": params["T2M"].get(timestamp),
            "precip_mm": params["PRECTOTCORR"].get(timestamp),
            "humidity_percent": params["RH2M"].get(timestamp),
            "wind_speed_mps": params["WS10M"].get(timestamp)
        })

    df = pd.DataFrame(rows)

    for col in ["temperature_c", "precip_mm", "humidity_percent", "wind_speed_mps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def classify_heat(max_temp):
    if max_temp >= THRESHOLDS["heat_critical_c"]:
        return "CRITICAL"
    elif max_temp >= THRESHOLDS["heat_warning_c"]:
        return "WARNING"
    elif max_temp >= THRESHOLDS["heat_watch_c"]:
        return "WATCH"
    else:
        return "NORMAL"


def classify_rain(rain_24h):
    if rain_24h >= THRESHOLDS["rain_critical_mm_24h"]:
        return "CRITICAL"
    elif rain_24h >= THRESHOLDS["rain_warning_mm_24h"]:
        return "WARNING"
    elif rain_24h >= THRESHOLDS["rain_watch_mm_24h"]:
        return "WATCH"
    else:
        return "NORMAL"


def heat_agent(df):
    max_temp = round(float(df["temperature_c"].max()), 2)
    mean_temp = round(float(df["temperature_c"].mean()), 2)
    risk = classify_heat(max_temp)

    return {
        "agent": "HeatAgent",
        "risk": risk,
        "max_temperature_c": max_temp,
        "mean_temperature_c": mean_temp,
        "summary": f"Maximum temperature recorded: {max_temp} C"
    }


def rainfall_agent(df):
    recent_24 = df.tail(24)
    rain_24h = round(float(recent_24["precip_mm"].sum()), 2)
    risk = classify_rain(rain_24h)

    return {
        "agent": "RainfallAgent",
        "risk": risk,
        "rainfall_24h_mm": rain_24h,
        "summary": f"Last 24-hour precipitation: {rain_24h} mm"
    }


def glacier_melt_proxy_agent(heat_result):
    """
    Early prototype proxy:
    If heat is high, glacier/snow melt potential increases.
    Later we will replace/add Sentinel-2, MODIS snow, and GEE.
    """

    heat_risk = heat_result["risk"]
    max_temp = heat_result["max_temperature_c"]

    if heat_risk == "CRITICAL":
        risk = "CRITICAL"
    elif heat_risk == "WARNING":
        risk = "WARNING"
    elif heat_risk == "WATCH":
        risk = "WATCH"
    else:
        risk = "NORMAL"

    return {
        "agent": "GlacierMeltProxyAgent",
        "risk": risk,
        "summary": f"Glacier melt proxy based on high temperature: {max_temp} C"
    }


def flood_proxy_agent(rain_result, glacier_result):
    """
    Prototype flood risk:
    Flood risk increases when rainfall and melt proxy are both active.
    """

    rain_points = RISK_POINTS.get(rain_result["risk"], 0)
    melt_points = RISK_POINTS.get(glacier_result["risk"], 0)

    combined = rain_points + melt_points

    if combined >= 5:
        risk = "CRITICAL"
    elif combined >= 3:
        risk = "WARNING"
    elif combined >= 1:
        risk = "WATCH"
    else:
        risk = "NORMAL"

    return {
        "agent": "FloodProxyAgent",
        "risk": risk,
        "summary": "Flood proxy based on rainfall plus glacier melt potential"
    }


def landslide_proxy_agent(rain_result):
    """
    Prototype landslide risk:
    Strong rainfall signal increases landslide potential.
    Later we will add slope, DEM, landcover and Sentinel-1.
    """

    rain_risk = rain_result["risk"]

    if rain_risk == "CRITICAL":
        risk = "CRITICAL"
    elif rain_risk == "WARNING":
        risk = "WARNING"
    elif rain_risk == "WATCH":
        risk = "WATCH"
    else:
        risk = "NORMAL"

    return {
        "agent": "LandslideProxyAgent",
        "risk": risk,
        "summary": "Landslide proxy based on recent rainfall intensity"
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
                "summary": result.get("summary", "")
            })

    if score >= 13:
        level = "CRITICAL"
    elif score >= 8:
        level = "WARNING"
    elif score >= 3:
        level = "WATCH"
    else:
        level = "NORMAL"

    return {
        "risk_level": level,
        "score": round(score, 2),
        "reasons": reasons
    }


def send_telegram_alert(result):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram token or chat ID missing. Skipping alert.")
        return

    combined = result["combined_risk"]

    if combined["risk_level"] not in ["WATCH", "WARNING", "CRITICAL"]:
        print("Risk normal. No Telegram alert sent.")
        return

    message = f"""
Northern Pakistan Glacial Risk Watch

Risk Level: {combined['risk_level']}
Risk Score: {combined['score']}
Time UTC: {result['timestamp']}

Top Risk Reasons:
"""

    for reason in combined["reasons"][:10]:
        message += f"\n- {reason['agent']} in {reason.get('valley', 'region')}: {reason['risk']} | {reason['summary']}"

    message += "\n\nNote: Automated decision-support only. Verify with official authorities."

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message
        },
        timeout=20
    )


def save_outputs(final_result):
    os.makedirs("data", exist_ok=True)

    with open("data/latest_result.json", "w") as f:
        json.dump(final_result, f, indent=2)

    history_path = "data/history.csv"

    row = {
        "timestamp": final_result["timestamp"],
        "risk_level": final_result["combined_risk"]["risk_level"],
        "score": final_result["combined_risk"]["score"],
        "valleys_monitored": len(final_result["valleys"])
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
            "score": valley["combined_risk"]["score"]
        })

    pd.DataFrame(valley_rows).to_csv("data/latest_valleys.csv", index=False)


def run_for_valley(valley):
    print(f"Running agents for {valley['name']}")

    df = fetch_nasa_power_hourly(valley["lat"], valley["lon"])

    heat = heat_agent(df)
    rain = rainfall_agent(df)
    glacier = glacier_melt_proxy_agent(heat)
    flood = flood_proxy_agent(rain, glacier)
    landslide = landslide_proxy_agent(rain)

    agent_results = [
        heat,
        rain,
        glacier,
        flood,
        landslide
    ]

    combined = risk_fusion_agent(agent_results)

    return {
        "name": valley["name"],
        "lat": valley["lat"],
        "lon": valley["lon"],
        "agents": agent_results,
        "combined_risk": combined
    }


def main():
    valley_results = []

    for valley in VALLEYS:
        try:
            valley_result = run_for_valley(valley)
            valley_results.append(valley_result)
        except Exception as e:
            valley_results.append({
                "name": valley["name"],
                "lat": valley["lat"],
                "lon": valley["lon"],
                "error": str(e),
                "agents": [],
                "combined_risk": {
                    "risk_level": "UNKNOWN",
                    "score": 0,
                    "reasons": []
                }
            })

    all_agent_results = []

    for valley in valley_results:
        for reason in valley["combined_risk"].get("reasons", []):
            reason["valley"] = valley["name"]

        all_agent_results.append({
            "agent": f"ValleyRisk_{valley['name']}",
            "risk": valley["combined_risk"]["risk_level"],
            "summary": f"{valley['name']} score {valley['combined_risk']['score']}"
        })

    overall = risk_fusion_agent(all_agent_results)

    final_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "Northern Pakistan Glacial Flood and Landslide Watch",
        "combined_risk": overall,
        "valleys": valley_results
    }

    save_outputs(final_result)
    send_telegram_alert(final_result)

    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()