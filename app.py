# ============================================================
# MINEVISION V6
# Intelligent Mine Vehicle Safety Decision-Support System
#
# Safe and Efficient Operation of Mine Vehicles
# in Fog and Low-Visibility Conditions
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import math
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MineVision V6",
    page_icon="⛏️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    font-size: 17px;
    color: #6b7280;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">⛏️ MineVision V6</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Safety Decision-Support System for Open-Cast Mines'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "vehicles" not in st.session_state:
    st.session_state.vehicles = []

if "running" not in st.session_state:
    st.session_state.running = False

if "step" not in st.session_state:
    st.session_state.step = 0

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []


# ============================================================
# ROAD NETWORK
# ============================================================

ROAD_POINTS = [

    (12, 82),
    (25, 82),
    (40, 72),
    (55, 72),
    (70, 82),
    (88, 82),

    (40, 72),
    (40, 52),
    (55, 52),
    (70, 52),

    (55, 72),
    (55, 52),
    (55, 30),

    (55, 30),
    (70, 22),
    (88, 22),

    (40, 52),
    (28, 38),
    (18, 25)

]


# ============================================================
# FUNCTIONS
# ============================================================

def get_fog_condition(visibility):

    if visibility < 30:
        return "HEAVY FOG"

    elif visibility < 60:
        return "MODERATE FOG"

    elif visibility < 100:
        return "LIGHT FOG"

    return "CLEAR"


# ------------------------------------------------------------

def calculate_speed_limit(visibility):

    if visibility < 30:
        return 10

    elif visibility < 60:
        return 20

    elif visibility < 100:
        return 30

    elif visibility < 150:
        return 40

    return 50


# ------------------------------------------------------------

def calculate_safe_distance(
    speed,
    visibility
):

    distance = speed * 2

    if visibility < 30:
        distance += 30

    elif visibility < 60:
        distance += 20

    elif visibility < 100:
        distance += 10

    return distance


# ------------------------------------------------------------

def get_status(risk):

    if risk >= 70:
        return "DANGER"

    elif risk >= 40:
        return "CAUTION"

    return "SAFE"


# ------------------------------------------------------------

def get_recommendation(
    status,
    speed,
    speed_limit,
    nearest_distance,
    safe_distance,
    restricted
):

    if status == "DANGER":

        if restricted:
            return "STOP - RESTRICTED ZONE"

        if nearest_distance < 30:
            return "STOP - PROXIMITY HAZARD"

        if speed > speed_limit:
            return "STOP - EXCESSIVE SPEED"

        return "STOP - HIGH OPERATIONAL RISK"


    if status == "CAUTION":

        if restricted:
            return "LEAVE RESTRICTED ZONE"

        if speed > speed_limit:
            return "REDUCE SPEED"

        if nearest_distance < safe_distance:
            return "INCREASE DISTANCE"

        return "PROCEED WITH CAUTION"


    return "NORMAL OPERATION"


# ============================================================
# RISK COMPONENTS
# ============================================================

def visibility_risk(visibility):

    if visibility < 30:
        return 35

    elif visibility < 60:
        return 25

    elif visibility < 100:
        return 15

    elif visibility < 150:
        return 5

    return 0


# ------------------------------------------------------------

def speed_risk(
    speed,
    speed_limit
):

    if speed > speed_limit * 1.5:
        return 30

    elif speed > speed_limit:
        return 25

    elif speed > speed_limit * 0.8:
        return 10

    return 0


# ------------------------------------------------------------

def distance_risk(
    nearest_distance,
    safe_distance
):

    if nearest_distance < 20:
        return 35

    elif nearest_distance < 30:
        return 30

    elif nearest_distance < safe_distance:
        return 20

    elif nearest_distance < safe_distance * 1.3:
        return 10

    return 0


# ------------------------------------------------------------

def zone_risk(restricted):

    if restricted:
        return 25

    return 0


# ------------------------------------------------------------

def traffic_risk(
    number_of_vehicles
):

    if number_of_vehicles >= 9:
        return 10

    elif number_of_vehicles >= 6:
        return 5

    return 0


# ============================================================
# CREATE VEHICLE
# ============================================================

def create_vehicle(number):

    starting_points = [

        (12, 82),
        (25, 82),
        (40, 72),
        (55, 72),
        (70, 82),
        (40, 52),
        (55, 30),
        (70, 22),
        (28, 38),
        (18, 25)

    ]

    point = starting_points[
        (number - 1)
        % len(starting_points)
    ]

    return {

        "id": f"T-{number:02d}",

        "x": point[0],

        "y": point[1],

        "speed": random.randint(
            15,
            40
        ),

        "target_index": random.randint(
            0,
            len(ROAD_POINTS) - 1
        )

    }


# ============================================================
# MOVE VEHICLE
# ============================================================

def move_vehicle(vehicle):

    target = ROAD_POINTS[
        vehicle["target_index"]
    ]

    dx = target[0] - vehicle["x"]

    dy = target[1] - vehicle["y"]

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    if distance < 2:

        vehicle["target_index"] = (

            vehicle["target_index"] + 1

        ) % len(ROAD_POINTS)

        return


    movement = random.uniform(
        0.8,
        1.8
    )

    vehicle["x"] += (
        dx / distance
    ) * movement

    vehicle["y"] += (
        dy / distance
    ) * movement


# ============================================================
# RESTRICTED ZONE
# ============================================================

def is_restricted_zone(
    x,
    y
):

    return (
        10 <= x <= 30
        and
        10 <= y <= 30
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Mine Control Center"
)


st.sidebar.subheader(
    "Environmental Conditions"
)


visibility = st.sidebar.slider(
    "🌫️ Visibility",
    min_value=20,
    max_value=200,
    value=120,
    step=5
)


vehicle_count = st.sidebar.slider(
    "🚛 Active Vehicles",
    min_value=3,
    max_value=10,
    value=6
)


st.sidebar.subheader(
    "Simulation"
)


simulation_speed = st.sidebar.slider(
    "Simulation Speed",
    min_value=0.1,
    max_value=1.0,
    value=0.4
)


st.sidebar.divider()


if st.sidebar.button(
    "▶️ Start / Resume"
):

    st.session_state.running = True


if st.sidebar.button(
    "⏸️ Pause"
):

    st.session_state.running = False


if st.sidebar.button(
    "🔄 Reset"
):

    st.session_state.vehicles = []

    st.session_state.step = 0

    st.session_state.initialized = False

    st.session_state.running = False

    st.session_state.risk_history = []

    st.session_state.alert_history = []


# ============================================================
# ENVIRONMENT
# ============================================================

fog_condition = get_fog_condition(
    visibility
)


speed_limit = calculate_speed_limit(
    visibility
)


# ============================================================
# INITIALIZE VEHICLES
# ============================================================

if (
    not st.session_state.initialized
    or
    len(st.session_state.vehicles)
    != vehicle_count
):

    st.session_state.vehicles = []

    for i in range(
        vehicle_count
    ):

        st.session_state.vehicles.append(
            create_vehicle(i + 1)
        )

    st.session_state.initialized = True


# ============================================================
# MOVE VEHICLES
# ============================================================

if st.session_state.running:

    for vehicle in st.session_state.vehicles:

        move_vehicle(vehicle)

        vehicle["speed"] = random.randint(

            max(
                8,
                speed_limit - 5
            ),

            min(
                55,
                speed_limit + 20
            )

        )

    st.session_state.step += 1


# ============================================================
# VEHICLE RISK ANALYSIS
# ============================================================

vehicle_data = []


for vehicle in st.session_state.vehicles:

    nearest_distance = 999

    nearest_vehicle = "None"


    # --------------------------------------------------------
    # Find nearest vehicle
    # --------------------------------------------------------

    for other in st.session_state.vehicles:

        if vehicle["id"] == other["id"]:
            continue


        dx = (
            vehicle["x"]
            - other["x"]
        )

        dy = (
            vehicle["y"]
            - other["y"]
        )


        distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )


        if distance < nearest_distance:

            nearest_distance = distance

            nearest_vehicle = other["id"]


    # Convert simulation units to meters

    real_distance = round(
        nearest_distance * 3,
        1
    )


    # Safe distance

    safe_distance = calculate_safe_distance(

        vehicle["speed"],

        visibility

    )


    # Restricted area

    restricted = is_restricted_zone(

        vehicle["x"],

        vehicle["y"]

    )


    # --------------------------------------------------------
    # Risk Components
    # --------------------------------------------------------

    v_risk = visibility_risk(
        visibility
    )


    s_risk = speed_risk(

        vehicle["speed"],

        speed_limit

    )


    d_risk = distance_risk(

        real_distance,

        safe_distance

    )


    z_risk = zone_risk(
        restricted
    )


    t_risk = traffic_risk(
        vehicle_count
    )


    # --------------------------------------------------------
    # Total Risk
    # --------------------------------------------------------

    total_risk = min(

        v_risk
        +
        s_risk
        +
        d_risk
        +
        z_risk
        +
        t_risk,

        100

    )


    status = get_status(
        total_risk
    )


    recommendation = get_recommendation(

        status,

        vehicle["speed"],

        speed_limit,

        real_distance,

        safe_distance,

        restricted

    )


    vehicle_data.append({

        "Vehicle": vehicle["id"],

        "Speed": vehicle["speed"],

        "Distance": real_distance,

        "Safe Distance": safe_distance,

        "Visibility Risk": v_risk,

        "Speed Risk": s_risk,

        "Distance Risk": d_risk,

        "Zone Risk": z_risk,

        "Traffic Risk": t_risk,

        "Risk": total_risk,

        "Status": status,

        "Recommendation": recommendation,

        "Nearest Vehicle": nearest_vehicle,

        "x": vehicle["x"],

        "y": vehicle["y"],

        "Restricted": restricted

    })


df = pd.DataFrame(
    vehicle_data
)


# ============================================================
# MINE SAFETY SCORE
# ============================================================

average_risk = round(
    df["Risk"].mean(),
    1
)


mine_safety_score = max(
    0,
    round(
        100 - average_risk,
        1
    )
)


# Store history

if st.session_state.step > 0:

    st.session_state.risk_history.append({

        "Step": st.session_state.step,

        "Risk": average_risk,

        "Safety Score": mine_safety_score

    })


# Keep history manageable

if len(
    st.session_state.risk_history
) > 50:

    st.session_state.risk_history = (
        st.session_state.risk_history[-50:]
    )


# ============================================================
# STATUS COUNTS
# ============================================================

safe_count = sum(
    df["Status"] == "SAFE"
)

caution_count = sum(
    df["Status"] == "CAUTION"
)

danger_count = sum(
    df["Status"] == "DANGER"
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    "📊 Mine Safety Overview"
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "🛡️ Safety Score",
        f"{mine_safety_score}/100"
    )


with col2:

    st.metric(
        "🌫️ Visibility",
        f"{visibility} m"
    )


with col3:

    st.metric(
        "⚡ Speed Limit",
        f"{speed_limit} km/h"
    )


with col4:

    st.metric(
        "🚛 Vehicles",
        vehicle_count
    )


with col5:

    st.metric(
        "🔴 Critical",
        danger_count
    )


st.divider()


# ============================================================
# MAIN MAP
# ============================================================

st.subheader(
    "🗺️ Intelligent Mine Operations Map"
)


fig = go.Figure()


# ------------------------------------------------------------
# Mine boundary
# ------------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=[
            5,
            95,
            95,
            5,
            5
        ],

        y=[
            5,
            5,
            95,
            95,
            5
        ],

        mode="lines",

        line=dict(
            width=4
        ),

        name="Mine Boundary",

        hoverinfo="skip"

    )

)


# ------------------------------------------------------------
# Roads
# ------------------------------------------------------------

road_x = []

road_y = []


for i in range(
    len(ROAD_POINTS) - 1
):

    x1, y1 = ROAD_POINTS[i]

    x2, y2 = ROAD_POINTS[i + 1]


    road_x.extend(
        [
            x1,
            x2,
            None
        ]
    )

    road_y.extend(
        [
            y1,
            y2,
            None
        ]
    )


fig.add_trace(

    go.Scatter(

        x=road_x,

        y=road_y,

        mode="lines",

        line=dict(
            width=10
        ),

        name="Haul Roads",

        hoverinfo="skip"

    )

)


# ------------------------------------------------------------
# Vehicle colors
# ------------------------------------------------------------

colors = []


for status in df["Status"]:

    if status == "DANGER":

        colors.append("red")

    elif status == "CAUTION":

        colors.append("orange")

    else:

        colors.append("green")


# ------------------------------------------------------------
# Vehicles
# ------------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=df["x"],

        y=df["y"],

        mode="markers+text",

        marker=dict(

            size=18,

            color=colors,

            line=dict(
                width=2
            )

        ),

        text=df["Vehicle"],

        textposition="top center",

        name="Vehicles",

        hovertemplate=(

            "<b>%{text}</b><br>"
            "Risk: %{customdata[0]}/100<br>"
            "Speed: %{customdata[1]} km/h<br>"
            "Distance: %{customdata[2]} m"
            "<extra></extra>"

        ),

        customdata=df[
            [
                "Risk",
                "Speed",
                "Distance"
            ]
        ].values

    )

)


# ------------------------------------------------------------
# Excavation Zone
# ------------------------------------------------------------

fig.add_shape(

    type="circle",

    x0=10,

    y0=65,

    x1=30,

    y1=85,

    line=dict(
        width=2
    ),

    fillcolor="rgba(120,120,120,0.25)"

)


fig.add_annotation(

    x=20,

    y=75,

    text="⛏️ EXCAVATION",

    showarrow=False

)


# ------------------------------------------------------------
# Loading Zone
# ------------------------------------------------------------

fig.add_shape(

    type="rect",

    x0=70,

    y0=70,

    x1=90,

    y1=90,

    line=dict(
        width=2
    ),

    fillcolor="rgba(100,100,100,0.25)"

)


fig.add_annotation(

    x=80,

    y=80,

    text="🏭 LOADING",

    showarrow=False

)


# ------------------------------------------------------------
# Dump Area
# ------------------------------------------------------------

fig.add_shape(

    type="rect",

    x0=70,

    y0=10,

    x1=92,

    y1=30,

    line=dict(
        width=2
    ),

    fillcolor="rgba(100,100,100,0.25)"

)


fig.add_annotation(

    x=81,

    y=20,

    text="🏗️ DUMP",

    showarrow=False

)


# ------------------------------------------------------------
# Restricted Zone
# ------------------------------------------------------------

fig.add_shape(

    type="rect",

    x0=10,

    y0=10,

    x1=30,

    y1=30,

    line=dict(

        width=3,

        dash="dash"

    ),

    fillcolor="rgba(255,0,0,0.08)"

)


fig.add_annotation(

    x=20,

    y=20,

    text="⚠️ RESTRICTED",

    showarrow=False

)


# ------------------------------------------------------------
# Fog Overlay
# ------------------------------------------------------------

fog_opacity = max(

    0.05,

    min(

        0.65,

        (200 - visibility) / 250

    )

)


fig.add_shape(

    type="rect",

    x0=5,

    y0=5,

    x1=95,

    y1=95,

    line=dict(
        width=0
    ),

    fillcolor=(

        f"rgba(180,180,180,"
        f"{fog_opacity})"

    )

)


# ------------------------------------------------------------
# Map Layout
# ------------------------------------------------------------

fig.update_layout(

    height=650,

    xaxis=dict(

        range=[
            0,
            100
        ],

        showgrid=False,

        zeroline=False,

        title=""

    ),

    yaxis=dict(

        range=[
            0,
            100
        ],

        showgrid=False,

        zeroline=False,

        title="",

        scaleanchor="x",

        scaleratio=1

    ),

    margin=dict(

        l=10,

        r=10,

        t=30,

        b=10

    )

)


st.plotly_chart(

    fig,

    use_container_width=True

)


# ============================================================
# SAFETY ALERTS
# ============================================================

st.subheader(
    "🚨 Intelligent Safety Alerts"
)


if danger_count > 0:

    st.error(

        f"CRITICAL ALERT: "
        f"{danger_count} vehicle(s) "
        f"have high operational risk."

    )


    danger_df = df[
        df["Status"] == "DANGER"
    ]


    for _, row in danger_df.iterrows():

        st.error(

            f"🚛 {row['Vehicle']} | "
            f"Risk: {row['Risk']}/100 | "
            f"{row['Recommendation']}"

        )


elif caution_count > 0:

    st.warning(

        f"CAUTION: "
        f"{caution_count} vehicle(s) "
        f"require attention."

    )


else:

    st.success(
        "ALL CLEAR — No critical risks detected."
    )


# ============================================================
# RISK COMPONENT ANALYSIS
# ============================================================

st.subheader(
    "🧠 Risk Component Analysis"
)


risk_components = pd.DataFrame({

    "Risk Factor": [

        "Visibility",

        "Speed",

        "Distance",

        "Restricted Zone",

        "Traffic"

    ],

    "Contribution": [

        df["Visibility Risk"].mean(),

        df["Speed Risk"].mean(),

        df["Distance Risk"].mean(),

        df["Zone Risk"].mean(),

        df["Traffic Risk"].mean()

    ]

})


st.bar_chart(

    risk_components.set_index(
        "Risk Factor"
    )

)


# ============================================================
# RISK HISTORY
# ============================================================

st.subheader(
    "📈 Mine Risk Trend"
)


if len(
    st.session_state.risk_history
) > 0:

    history_df = pd.DataFrame(

        st.session_state.risk_history

    )


    history_df = history_df.set_index(
        "Step"
    )


    st.line_chart(

        history_df[
            [
                "Risk",
                "Safety Score"
            ]
        ]

    )

else:

    st.info(
        "Start the simulation to generate "
        "historical safety data."
    )


# ============================================================
# VEHICLE MONITORING
# ============================================================

st.subheader(
    "🚛 Vehicle Intelligence Table"
)


display_df = df[

    [

        "Vehicle",

        "Speed",

        "Distance",

        "Safe Distance",

        "Risk",

        "Status",

        "Nearest Vehicle",

        "Recommendation"

    ]

].copy()


display_df.columns = [

    "Vehicle",

    "Speed (km/h)",

    "Nearest Vehicle (m)",

    "Required Distance (m)",

    "Risk Score",

    "Status",

    "Nearest Truck",

    "Recommendation"

]


st.dataframe(

    display_df,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# SAFETY DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Vehicle Safety Distribution"
)


distribution = pd.DataFrame({

    "Status": [

        "SAFE",

        "CAUTION",

        "DANGER"

    ],

    "Vehicles": [

        safe_count,

        caution_count,

        danger_count

    ]

})


st.bar_chart(

    distribution.set_index(
        "Status"
    )

)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

st.subheader(
    "🛡️ MineVision Recommendation Engine"
)


if visibility < 30:

    st.error(

        "HEAVY FOG DETECTED: "
        "Restrict vehicle movement. "
        "Use minimum operating speeds and "
        "maximum vehicle separation."

    )

elif visibility < 60:

    st.warning(

        "MODERATE FOG DETECTED: "
        "Reduce vehicle speeds and increase "
        "following distances."

    )

elif visibility < 100:

    st.info(

        "LIGHT FOG DETECTED: "
        "Increase monitoring frequency and "
        "maintain safe vehicle spacing."

    )

else:

    st.success(

        "VISIBILITY NORMAL: "
        "Continue standard safety monitoring."

    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()


st.subheader(
    "📡 MineVision System Status"
)


status1, status2, status3, status4 = st.columns(4)


with status1:

    if st.session_state.running:

        st.success(
            "SYSTEM RUNNING"
        )

    else:

        st.info(
            "SYSTEM PAUSED"
        )


with status2:

    st.write(
        f"**Simulation Step:** "
        f"{st.session_state.step}"
    )


with status3:

    st.write(
        f"**Vehicles Monitored:** "
        f"{vehicle_count}"
    )


with status4:

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    st.write(
        f"**Last Update:** "
        f"{current_time}"
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

with st.expander(
    "🧠 How MineVision Calculates Risk"
):

    st.write(
        """
        MineVision uses an explainable weighted-risk approach.

        The risk score considers:

        1. Visibility conditions
        2. Vehicle speed
        3. Distance from nearby vehicles
        4. Restricted-zone presence
        5. Traffic density

        Each factor contributes to the overall risk score.

        The resulting score is classified as:

        0–39   → SAFE

        40–69  → CAUTION

        70–100 → DANGER

        The system then produces an operational
        recommendation for the vehicle.
        """
    )


# ============================================================
# FUTURE INTEGRATION
# ============================================================

with st.expander(
    "🚀 Future Real-World Integration"
):

    st.write(
        """
        The prototype can be extended to receive:

        • GPS vehicle positions
        • Vehicle speed telemetry
        • Visibility sensors
        • Camera feeds
        • Weather data
        • Mine fleet-management systems
        • Edge/IoT sensor data

        These data sources could replace the current
        simulated inputs.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "MineVision V6 | Intelligent Mine Safety Prototype"
)


st.caption(
    "All environmental and vehicle data are simulated "
    "for demonstration purposes."
)


# ============================================================
# AUTOMATIC REFRESH
# ============================================================

if st.session_state.running:

    st.rerun()
