# ============================================================
# MINEVISION V7
# Predictive AI Mine Vehicle Safety System
#
# Safe and Efficient Operation of Mine Vehicles
# in Fog and Low-Visibility Conditions
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MineVision V7",
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
    margin-bottom: 20px;
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
    '<div class="title">⛏️ MineVision V7</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predictive AI Safety System for Open-Cast Mine Vehicles'
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
# MINE ROAD NETWORK
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
# HELPER FUNCTIONS
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

def is_restricted_zone(x, y):

    return (
        10 <= x <= 30
        and
        10 <= y <= 30
    )


# ============================================================
# GENERATE TRAINING DATA
# ============================================================

def generate_training_data(samples=5000):

    data = []

    for _ in range(samples):

        visibility = random.randint(
            20,
            200
        )

        speed = random.randint(
            5,
            60
        )

        distance = random.randint(
            10,
            180
        )

        traffic = random.randint(
            2,
            10
        )

        restricted = random.randint(
            0,
            1
        )

        speed_limit = calculate_speed_limit(
            visibility
        )

        safe_distance = calculate_safe_distance(
            speed,
            visibility
        )

        # --------------------------------------------
        # Generate training risk
        # --------------------------------------------

        risk = 0

        # Visibility

        if visibility < 30:
            risk += 35

        elif visibility < 60:
            risk += 25

        elif visibility < 100:
            risk += 15

        elif visibility < 150:
            risk += 5


        # Speed

        if speed > speed_limit * 1.5:
            risk += 30

        elif speed > speed_limit:
            risk += 25

        elif speed > speed_limit * 0.8:
            risk += 10


        # Distance

        if distance < 20:
            risk += 35

        elif distance < 30:
            risk += 30

        elif distance < safe_distance:
            risk += 20

        elif distance < safe_distance * 1.3:
            risk += 10


        # Restricted zone

        if restricted:
            risk += 25


        # Traffic

        if traffic >= 9:
            risk += 10

        elif traffic >= 6:
            risk += 5


        # Small random noise makes the dataset
        # less perfectly deterministic.

        risk += random.randint(
            -5,
            5
        )

        risk = max(
            0,
            min(
                risk,
                100
            )
        )


        # --------------------------------------------
        # Convert risk into class
        # --------------------------------------------

        if risk >= 70:

            label = "DANGER"

        elif risk >= 40:

            label = "CAUTION"

        else:

            label = "SAFE"


        data.append({

            "visibility": visibility,

            "speed": speed,

            "distance": distance,

            "traffic": traffic,

            "restricted": restricted,

            "risk": risk,

            "label": label

        })


    return pd.DataFrame(data)


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model():

    training_data = generate_training_data(
        5000
    )


    features = [

        "visibility",

        "speed",

        "distance",

        "traffic",

        "restricted"

    ]


    X = training_data[
        features
    ]

    y = training_data[
        "label"
    ]


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    model = RandomForestClassifier(

        n_estimators=150,

        max_depth=10,

        random_state=42

    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return (
        model,
        training_data,
        accuracy
    )


# ============================================================
# LOAD MODEL
# ============================================================

model, training_data, model_accuracy = train_model()


FEATURES = [

    "visibility",

    "speed",

    "distance",

    "traffic",

    "restricted"

]


# ============================================================
# VEHICLE CREATION
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
        %
        len(starting_points)
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


    dx = (
        target[0]
        -
        vehicle["x"]
    )

    dy = (
        target[1]
        -
        vehicle["y"]
    )


    distance = math.sqrt(
        dx * dx
        +
        dy * dy
    )


    if distance < 2:

        vehicle["target_index"] = (

            vehicle["target_index"]
            +
            1

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

    len(
        st.session_state.vehicles
    )
    !=
    vehicle_count

):

    st.session_state.vehicles = []


    for i in range(
        vehicle_count
    ):

        st.session_state.vehicles.append(

            create_vehicle(
                i + 1
            )

        )


    st.session_state.initialized = True


# ============================================================
# MOVE VEHICLES
# ============================================================

if st.session_state.running:

    for vehicle in st.session_state.vehicles:

        move_vehicle(
            vehicle
        )


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
# BUILD VEHICLE DATA
# ============================================================

vehicle_data = []


for vehicle in st.session_state.vehicles:

    nearest_distance = 999

    nearest_vehicle = "None"


    # --------------------------------------------------------
    # Find nearest vehicle
    # --------------------------------------------------------

    for other in st.session_state.vehicles:

        if (
            vehicle["id"]
            ==
            other["id"]
        ):
            continue


        dx = (

            vehicle["x"]
            -
            other["x"]

        )


        dy = (

            vehicle["y"]
            -
            other["y"]

        )


        distance = math.sqrt(

            dx * dx
            +
            dy * dy

        )


        if distance < nearest_distance:

            nearest_distance = distance

            nearest_vehicle = other["id"]


    # Convert map units to meters

    real_distance = round(

        nearest_distance * 3,

        1

    )


    safe_distance = calculate_safe_distance(

        vehicle["speed"],

        visibility

    )


    restricted = is_restricted_zone(

        vehicle["x"],

        vehicle["y"]

    )


    # --------------------------------------------------------
    # MACHINE LEARNING INPUT
    # --------------------------------------------------------

    model_input = pd.DataFrame([{

        "visibility": visibility,

        "speed": vehicle["speed"],

        "distance": real_distance,

        "traffic": vehicle_count,

        "restricted": int(
            restricted
        )

    }])


    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        model_input
    )[0]


    probabilities = model.predict_proba(
        model_input
    )[0]


    confidence = round(

        max(
            probabilities
        )
        *
        100,

        1

    )


    # --------------------------------------------------------
    # Calculate numerical risk
    # --------------------------------------------------------

    risk_mapping = {

        "SAFE": 20,

        "CAUTION": 55,

        "DANGER": 90

    }


    predicted_risk = risk_mapping[
        prediction
    ]


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if prediction == "DANGER":

        if restricted:

            recommendation = (
                "STOP - RESTRICTED ZONE"
            )

        elif real_distance < 30:

            recommendation = (
                "STOP - PROXIMITY HAZARD"
            )

        elif vehicle["speed"] > speed_limit:

            recommendation = (
                "STOP - EXCESSIVE SPEED"
            )

        else:

            recommendation = (
                "STOP - HIGH PREDICTED RISK"
            )


    elif prediction == "CAUTION":

        if vehicle["speed"] > speed_limit:

            recommendation = (
                "REDUCE SPEED"
            )

        elif real_distance < safe_distance:

            recommendation = (
                "INCREASE VEHICLE DISTANCE"
            )

        else:

            recommendation = (
                "PROCEED WITH CAUTION"
            )


    else:

        recommendation = (
            "NORMAL OPERATION"
        )


    vehicle_data.append({

        "Vehicle": vehicle["id"],

        "Speed": vehicle["speed"],

        "Distance": real_distance,

        "Safe Distance": safe_distance,

        "Prediction": prediction,

        "Confidence": confidence,

        "Risk": predicted_risk,

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
# SAFETY SCORE
# ============================================================

average_risk = round(
    df["Risk"].mean(),
    1
)


safety_score = max(

    0,

    round(
        100 - average_risk,
        1
    )

)


# ============================================================
# COUNTERS
# ============================================================

safe_count = sum(
    df["Prediction"] == "SAFE"
)


caution_count = sum(
    df["Prediction"] == "CAUTION"
)


danger_count = sum(
    df["Prediction"] == "DANGER"
)


# ============================================================
# SAVE HISTORY
# ============================================================

if st.session_state.step > 0:

    st.session_state.risk_history.append({

        "Step": st.session_state.step,

        "Predicted Risk": average_risk,

        "Safety Score": safety_score

    })


if len(
    st.session_state.risk_history
) > 50:

    st.session_state.risk_history = (

        st.session_state.risk_history[-50:]

    )


# ============================================================
# TOP DASHBOARD
# ============================================================

st.subheader(
    "🤖 Predictive Safety Dashboard"
)


c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.metric(

        "🛡️ Safety Score",

        f"{safety_score}/100"

    )


with c2:

    st.metric(

        "🤖 ML Accuracy",

        f"{model_accuracy * 100:.1f}%"

    )


with c3:

    st.metric(

        "🌫️ Visibility",

        f"{visibility} m",

        fog_condition

    )


with c4:

    st.metric(

        "🚛 Vehicles",

        vehicle_count

    )


with c5:

    st.metric(

        "🔴 Danger",

        danger_count

    )


st.divider()


# ============================================================
# MINE MAP
# ============================================================

st.subheader(
    "🗺️ Predictive Mine Operations Map"
)


fig = go.Figure()


# ------------------------------------------------------------
# Mine Boundary
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


    road_x.extend([
        x1,
        x2,
        None
    ])


    road_y.extend([
        y1,
        y2,
        None
    ])


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
# Vehicle Colors
# ------------------------------------------------------------

vehicle_colors = []


for prediction in df[
    "Prediction"
]:

    if prediction == "DANGER":

        vehicle_colors.append(
            "red"
        )

    elif prediction == "CAUTION":

        vehicle_colors.append(
            "orange"
        )

    else:

        vehicle_colors.append(
            "green"
        )


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

            color=vehicle_colors,

            line=dict(
                width=2
            )

        ),

        text=df["Vehicle"],

        textposition="top center",

        name="Mine Vehicles",

        customdata=df[
            [
                "Prediction",
                "Confidence",
                "Speed",
                "Distance"
            ]
        ].values,

        hovertemplate=(

            "<b>%{text}</b><br>"
            "Prediction: %{customdata[0]}<br>"
            "Confidence: %{customdata[1]}%<br>"
            "Speed: %{customdata[2]} km/h<br>"
            "Distance: %{customdata[3]} m"
            "<extra></extra>"

        )

    )

)


# ------------------------------------------------------------
# Excavation
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
# Loading
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
# Dump
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
# Restricted Area
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

        zeroline=False

    ),

    yaxis=dict(

        range=[
            0,
            100
        ],

        showgrid=False,

        zeroline=False,

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
# PREDICTIVE ALERTS
# ============================================================

st.subheader(
    "🚨 AI Safety Predictions"
)


if danger_count > 0:

    st.error(

        f"AI PREDICTION: "
        f"{danger_count} vehicle(s) "
        f"are currently classified as HIGH RISK."

    )


    danger_df = df[
        df["Prediction"] == "DANGER"
    ]


    for _, row in danger_df.iterrows():

        st.error(

            f"🚛 {row['Vehicle']} | "
            f"Prediction: DANGER | "
            f"Confidence: {row['Confidence']}% | "
            f"{row['Recommendation']}"

        )


elif caution_count > 0:

    st.warning(

        f"AI PREDICTION: "
        f"{caution_count} vehicle(s) "
        f"require additional attention."

    )


else:

    st.success(

        "AI PREDICTION: "
        "All monitored vehicles are currently "
        "within the SAFE class."

    )


# ============================================================
# VEHICLE TABLE
# ============================================================

st.subheader(
    "🚛 AI Vehicle Monitoring"
)


display_df = df[

    [

        "Vehicle",

        "Speed",

        "Distance",

        "Safe Distance",

        "Prediction",

        "Confidence",

        "Risk",

        "Nearest Vehicle",

        "Recommendation"

    ]

].copy()


display_df.columns = [

    "Vehicle",

    "Speed (km/h)",

    "Nearest Vehicle (m)",

    "Required Distance (m)",

    "AI Prediction",

    "Confidence",

    "Risk Score",

    "Nearest Truck",

    "Recommendation"

]


st.dataframe(

    display_df,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🧠 AI Feature Importance"
)


importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": model.feature_importances_

})


importance_df = importance_df.sort_values(

    "Importance",

    ascending=False

)


st.bar_chart(

    importance_df.set_index(
        "Feature"
    )

)


# ============================================================
# RISK TREND
# ============================================================

st.subheader(
    "📈 Predictive Risk Trend"
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

        history_df

    )

else:

    st.info(

        "Start the simulation to generate "
        "predictive risk history."

    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "🤖 AI Model Information"
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "Training Samples",
        len(training_data)
    )


with m2:

    st.metric(
        "Features",
        len(FEATURES)
    )


with m3:

    st.metric(
        "Model Accuracy",
        f"{model_accuracy * 100:.1f}%"
    )


# ============================================================
# TRAINING DATA PREVIEW
# ============================================================

with st.expander(
    "📚 View AI Training Dataset"
):

    st.write(
        """
        MineVision V7 currently uses a synthetic
        training dataset for demonstration.

        In a real deployment, these records would be
        replaced with historical mine-operation data.
        """
    )

    st.dataframe(

        training_data.head(20),

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# EXPLAINABILITY
# ============================================================

with st.expander(
    "🔍 How the AI Makes a Prediction"
):

    st.write(
        """
        MineVision V7 uses a Random Forest classifier.

        The model receives five features:

        • Visibility
        • Vehicle speed
        • Distance from the nearest vehicle
        • Traffic density
        • Restricted-zone status

        It then predicts one of three classes:

        SAFE
        CAUTION
        DANGER

        The confidence value represents the model's
        predicted probability for the selected class.

        Feature importance shows which input variables
        contributed most strongly to the trained model.
        """
    )


# ============================================================
# REAL-WORLD INTEGRATION
# ============================================================

with st.expander(
    "🚀 Real-World Deployment Path"
):

    st.write(
        """
        The current prototype uses simulated data.

        A production MineVision system could replace
        the simulated inputs with:

        • GPS
        • Vehicle telemetry
        • Visibility sensors
        • Weather sensors
        • Camera systems
        • Mine fleet-management systems
        • IoT devices

        Historical operational records could then be used
        to retrain the model on actual mine conditions.
        """
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.subheader(
    "📡 System Status"
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    if st.session_state.running:

        st.success(
            "AI MONITORING ACTIVE"
        )

    else:

        st.info(
            "SYSTEM PAUSED"
        )


with s2:

    st.write(
        f"**Simulation Step:** "
        f"{st.session_state.step}"
    )


with s3:

    st.write(
        f"**Vehicles:** "
        f"{vehicle_count}"
    )


with s4:

    st.write(
        "**Model:** Random Forest"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "MineVision V7 | Predictive AI Mine Safety Prototype"
)


st.caption(
    "Training and operational data are simulated "
    "for hackathon demonstration purposes."
)


# ============================================================
# REFRESH
# ============================================================

if st.session_state.running:

    st.rerun()
