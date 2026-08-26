# ============================================================
# MINEVISION V10
# AI-ASSISTED MINE VEHICLE SAFETY SYSTEM
#
# Safe and Efficient Operation of Mine Vehicles
# in Fog and Low-Visibility Conditions
# in Open Cast Iron Ore Mines
#
# HACKATHON PROTOTYPE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import math
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# OPTIONAL YOLO
# ============================================================

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MineVision V10",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    .status-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: white;
    }

    .alert-box {
        padding: 18px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .small-text {
        font-size: 13px;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⛏️ MineVision V10</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Assisted Predictive Safety for Mine Vehicles '
    'in Fog and Low-Visibility Conditions'
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

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "step" not in st.session_state:
    st.session_state.step = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "camera_image" not in st.session_state:
    st.session_state.camera_image = None


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
    (70, 22),
    (88, 22),
    (28, 38),
    (18, 25)
]


# ============================================================
# ENVIRONMENT FUNCTIONS
# ============================================================

def fog_condition(visibility):

    if visibility < 30:
        return "HEAVY FOG"

    elif visibility < 60:
        return "MODERATE FOG"

    elif visibility < 100:
        return "LIGHT FOG"

    else:
        return "CLEAR"


def speed_limit(visibility):

    if visibility < 30:
        return 10

    elif visibility < 60:
        return 20

    elif visibility < 100:
        return 30

    elif visibility < 150:
        return 40

    return 50


def safe_distance(speed, visibility):

    distance = speed * 2

    if visibility < 30:
        distance += 30

    elif visibility < 60:
        distance += 20

    elif visibility < 100:
        distance += 10

    return distance


# ============================================================
# TTC
# ============================================================

def calculate_ttc(distance, speed_a, speed_b):

    relative_speed = abs(speed_a - speed_b)

    if relative_speed <= 0:
        return float("inf")

    relative_speed_mps = relative_speed / 3.6

    return round(
        distance / relative_speed_mps,
        2
    )


# ============================================================
# TRAINING DATA
# ============================================================

def create_training_data(samples=7000):

    rows = []

    for _ in range(samples):

        visibility = random.randint(20, 200)

        speed = random.randint(5, 65)

        distance = random.randint(10, 200)

        traffic = random.randint(2, 10)

        restricted = random.randint(0, 1)

        ttc = random.uniform(1, 40)

        limit = speed_limit(visibility)

        required_distance = safe_distance(
            speed,
            visibility
        )

        risk = 0

        # Visibility risk

        if visibility < 30:
            risk += 30

        elif visibility < 60:
            risk += 22

        elif visibility < 100:
            risk += 12

        elif visibility < 150:
            risk += 5

        # Speed risk

        if speed > limit * 1.5:
            risk += 30

        elif speed > limit:
            risk += 20

        elif speed > limit * 0.8:
            risk += 8

        # Distance risk

        if distance < 20:
            risk += 30

        elif distance < required_distance:
            risk += 22

        elif distance < required_distance * 1.3:
            risk += 10

        # TTC risk

        if ttc < 3:
            risk += 35

        elif ttc < 5:
            risk += 25

        elif ttc < 8:
            risk += 15

        elif ttc < 12:
            risk += 5

        # Traffic risk

        if traffic >= 9:
            risk += 10

        elif traffic >= 6:
            risk += 5

        # Restricted area

        if restricted:
            risk += 20

        risk += random.randint(-5, 5)

        risk = max(
            0,
            min(
                100,
                risk
            )
        )

        if risk >= 70:
            label = "DANGER"

        elif risk >= 40:
            label = "CAUTION"

        else:
            label = "SAFE"

        rows.append({
            "visibility": visibility,
            "speed": speed,
            "distance": distance,
            "traffic": traffic,
            "restricted": restricted,
            "ttc": ttc,
            "risk": risk,
            "label": label
        })

    return pd.DataFrame(rows)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = create_training_data()

    features = [
        "visibility",
        "speed",
        "distance",
        "traffic",
        "restricted",
        "ttc"
    ]

    X = data[features]

    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
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

    return model, data, accuracy


model, training_data, model_accuracy = train_model()


FEATURES = [
    "visibility",
    "speed",
    "distance",
    "traffic",
    "restricted",
    "ttc"
]


# ============================================================
# VEHICLE GENERATION
# ============================================================

def create_vehicle(number):

    point = ROAD_POINTS[
        (number - 1)
        %
        len(ROAD_POINTS)
    ]

    return {
        "id": f"T-{number:02d}",
        "x": point[0],
        "y": point[1],
        "speed": random.randint(10, 40),
        "target": random.randint(
            0,
            len(ROAD_POINTS) - 1
        )
    }


# ============================================================
# VEHICLE MOVEMENT
# ============================================================

def move_vehicle(vehicle):

    target = ROAD_POINTS[
        vehicle["target"]
    ]

    dx = target[0] - vehicle["x"]

    dy = target[1] - vehicle["y"]

    distance = math.sqrt(
        dx ** 2 + dy ** 2
    )

    if distance < 2:

        vehicle["target"] = (
            vehicle["target"] + 1
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
# INITIALIZE VEHICLES
# ============================================================

vehicle_count = st.sidebar.slider(
    "🚛 Number of Vehicles",
    3,
    10,
    6
)


if (
    not st.session_state.initialized
    or
    len(st.session_state.vehicles)
    != vehicle_count
):

    st.session_state.vehicles = [
        create_vehicle(i + 1)
        for i in range(vehicle_count)
    ]

    st.session_state.initialized = True


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Mine Control"
)

st.sidebar.divider()

st.sidebar.subheader(
    "🌫️ Environment"
)

visibility = st.sidebar.slider(
    "Visibility (meters)",
    20,
    200,
    120,
    5
)

traffic_level = st.sidebar.slider(
    "Traffic Level",
    1,
    10,
    vehicle_count
)

st.sidebar.divider()

st.sidebar.subheader(
    "🎮 Simulation"
)

if st.sidebar.button(
    "▶️ Start Monitoring",
    use_container_width=True
):

    st.session_state.running = True


if st.sidebar.button(
    "⏸️ Pause Monitoring",
    use_container_width=True
):

    st.session_state.running = False


if st.sidebar.button(
    "🔄 Reset System",
    use_container_width=True
):

    st.session_state.running = False

    st.session_state.step = 0

    st.session_state.history = []

    st.session_state.vehicles = [
        create_vehicle(i + 1)
        for i in range(vehicle_count)
    ]


# ============================================================
# CAMERA SECTION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📷 Vision System"
)

vision_mode = st.sidebar.selectbox(
    "Vision Input",
    [
        "Disabled",
        "Upload Image",
        "Camera"
    ]
)


uploaded_file = None


if vision_mode == "Upload Image":

    uploaded_file = st.sidebar.file_uploader(
        "Upload road image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# ============================================================
# ENVIRONMENT STATUS
# ============================================================

condition = fog_condition(
    visibility
)

current_speed_limit = speed_limit(
    visibility
)


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
                current_speed_limit - 5
            ),
            min(
                60,
                current_speed_limit + 20
            )
        )

    st.session_state.step += 1


# ============================================================
# VEHICLE RISK ANALYSIS
# ============================================================

results = []


for vehicle in st.session_state.vehicles:

    nearest_distance = 999

    nearest_speed = 0

    nearest_id = "None"


    for other in st.session_state.vehicles:

        if vehicle["id"] == other["id"]:
            continue

        dx = vehicle["x"] - other["x"]

        dy = vehicle["y"] - other["y"]

        distance = math.sqrt(
            dx ** 2 + dy ** 2
        )

        if distance < nearest_distance:

            nearest_distance = distance

            nearest_speed = other["speed"]

            nearest_id = other["id"]


    estimated_distance = round(
        nearest_distance * 3,
        1
    )


    required_distance = safe_distance(
        vehicle["speed"],
        visibility
    )


    restricted = (
        10 <= vehicle["x"] <= 30
        and
        10 <= vehicle["y"] <= 30
    )


    ttc = calculate_ttc(
        estimated_distance,
        vehicle["speed"],
        nearest_speed
    )


    model_ttc = min(
        ttc,
        60
    )


    input_data = pd.DataFrame([{

        "visibility": visibility,

        "speed": vehicle["speed"],

        "distance": estimated_distance,

        "traffic": traffic_level,

        "restricted": int(
            restricted
        ),

        "ttc": model_ttc

    }])


    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0]


    confidence = round(
        max(probability) * 100,
        1
    )


    risk_score = {
        "SAFE": 20,
        "CAUTION": 55,
        "DANGER": 90
    }[prediction]


    reasons = []


    if visibility < 60:
        reasons.append(
            "low visibility"
        )


    if vehicle["speed"] > current_speed_limit:
        reasons.append(
            "speed exceeds recommended limit"
        )


    if estimated_distance < required_distance:
        reasons.append(
            "insufficient vehicle separation"
        )


    if ttc < 5:
        reasons.append(
            "low TTC"
        )


    if restricted:
        reasons.append(
            "restricted zone"
        )


    if not reasons:
        reasons.append(
            "conditions within monitored range"
        )


    if prediction == "DANGER":

        recommendation = (
            "STOP / ALERT OPERATOR"
        )

    elif prediction == "CAUTION":

        recommendation = (
            "REDUCE SPEED / INCREASE DISTANCE"
        )

    else:

        recommendation = (
            "CONTINUE MONITORED OPERATION"
        )


    results.append({

        "Vehicle": vehicle["id"],

        "Speed": vehicle["speed"],

        "Distance": estimated_distance,

        "Required Distance": required_distance,

        "TTC": ttc,

        "Risk": risk_score,

        "Prediction": prediction,

        "Confidence": confidence,

        "Reason": ", ".join(reasons),

        "Recommendation": recommendation,

        "Nearest Vehicle": nearest_id,

        "x": vehicle["x"],

        "y": vehicle["y"]

    })


df = pd.DataFrame(
    results
)


# ============================================================
# OVERALL METRICS
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
# HISTORY
# ============================================================

if st.session_state.running:

    st.session_state.history.append({

        "Step": st.session_state.step,

        "Risk": average_risk,

        "Safety Score": safety_score

    })


if len(
    st.session_state.history
) > 60:

    st.session_state.history = (
        st.session_state.history[-60:]
    )


# ============================================================
# TOP DASHBOARD
# ============================================================

st.subheader(
    "📊 Live Safety Overview"
)

m1, m2, m3, m4, m5 = st.columns(5)


with m1:

    st.metric(
        "🛡️ Safety Score",
        f"{safety_score}/100"
    )


with m2:

    st.metric(
        "🌫️ Visibility",
        f"{visibility} m",
        condition
    )


with m3:

    st.metric(
        "🚛 Vehicles",
        vehicle_count
    )


with m4:

    st.metric(
        "⚠️ Caution",
        caution_count
    )


with m5:

    st.metric(
        "🚨 Danger",
        danger_count
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

if danger_count > 0:

    st.error(
        f"🚨 CRITICAL ALERT — "
        f"{danger_count} vehicle(s) classified as DANGER."
    )

elif caution_count > 0:

    st.warning(
        f"⚠️ CAUTION — "
        f"{caution_count} vehicle(s) require attention."
    )

else:

    st.success(
        "✅ SYSTEM STATUS — "
        "All monitored vehicles are currently SAFE."
    )


# ============================================================
# MINE MAP
# ============================================================

st.divider()

st.subheader(
    "🗺️ Live Mine Digital Map"
)


fig = go.Figure()


# ------------------------------------------------------------
# Mine boundary
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=[5, 95, 95, 5, 5],
        y=[5, 5, 95, 95, 5],
        mode="lines",
        line=dict(width=4),
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
        line=dict(width=10),
        name="Haul Roads",
        hoverinfo="skip"
    )
)


# ------------------------------------------------------------
# Vehicle colors
# ------------------------------------------------------------

vehicle_colors = []


for prediction in df["Prediction"]:

    if prediction == "DANGER":

        vehicle_colors.append("red")

    elif prediction == "CAUTION":

        vehicle_colors.append("orange")

    else:

        vehicle_colors.append("green")


# ------------------------------------------------------------
# Vehicles
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers+text",
        marker=dict(
            size=20,
            color=vehicle_colors,
            line=dict(width=2)
        ),
        text=df["Vehicle"],
        textposition="top center",
        name="Vehicles",
        customdata=df[
            [
                "Prediction",
                "Speed",
                "Distance",
                "TTC",
                "Confidence"
            ]
        ].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Risk: %{customdata[0]}<br>"
            "Speed: %{customdata[1]} km/h<br>"
            "Distance: %{customdata[2]} m<br>"
            "TTC: %{customdata[3]} s<br>"
            "Confidence: %{customdata[4]}%"
            "<extra></extra>"
        )
    )
)


# ------------------------------------------------------------
# Restricted zone
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
# Excavation area
# ------------------------------------------------------------

fig.add_shape(
    type="circle",
    x0=10,
    y0=65,
    x1=30,
    y1=85,
    line=dict(width=2),
    fillcolor="rgba(100,100,100,0.2)"
)


fig.add_annotation(
    x=20,
    y=75,
    text="⛏️ EXCAVATION",
    showarrow=False
)


# ------------------------------------------------------------
# Loading zone
# ------------------------------------------------------------

fig.add_shape(
    type="rect",
    x0=70,
    y0=70,
    x1=90,
    y1=90,
    line=dict(width=2),
    fillcolor="rgba(100,100,100,0.2)"
)


fig.add_annotation(
    x=80,
    y=80,
    text="🏭 LOADING",
    showarrow=False
)


# ------------------------------------------------------------
# Dump zone
# ------------------------------------------------------------

fig.add_shape(
    type="rect",
    x0=70,
    y0=10,
    x1=92,
    y1=30,
    line=dict(width=2),
    fillcolor="rgba(100,100,100,0.2)"
)


fig.add_annotation(
    x=81,
    y=20,
    text="🏗️ DUMP",
    showarrow=False
)


# ------------------------------------------------------------
# Fog overlay
# ------------------------------------------------------------

opacity = max(
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
    line=dict(width=0),
    fillcolor=(
        f"rgba(180,180,180,{opacity})"
    )
)


fig.update_layout(

    height=650,

    xaxis=dict(
        range=[0, 100],
        showgrid=False,
        title=""
    ),

    yaxis=dict(
        range=[0, 100],
        showgrid=False,
        title="",
        scaleanchor="x",
        scaleratio=1
    ),

    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TWO-COLUMN ANALYSIS
# ============================================================

left, right = st.columns(2)


# ============================================================
# LEFT — VEHICLE INTELLIGENCE
# ============================================================

with left:

    st.subheader(
        "🚛 Vehicle Intelligence"
    )

    display_df = df[
        [
            "Vehicle",
            "Speed",
            "Distance",
            "Required Distance",
            "TTC",
            "Prediction",
            "Confidence"
        ]
    ].copy()

    display_df.columns = [
        "Vehicle",
        "Speed km/h",
        "Distance m",
        "Required m",
        "TTC sec",
        "AI Risk",
        "Confidence"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RIGHT — ALERT CENTER
# ============================================================

with right:

    st.subheader(
        "🚨 Alert Center"
    )

    danger_df = df[
        df["Prediction"] != "SAFE"
    ]


    if len(danger_df) == 0:

        st.success(
            "No active vehicle safety alerts."
        )

    else:

        for _, row in danger_df.iterrows():

            if row["Prediction"] == "DANGER":

                st.error(
                    f"🚛 {row['Vehicle']} — "
                    f"DANGER\n\n"
                    f"{row['Reason']}\n\n"
                    f"Action: {row['Recommendation']}"
                )

            else:

                st.warning(
                    f"🚛 {row['Vehicle']} — "
                    f"CAUTION\n\n"
                    f"{row['Reason']}\n\n"
                    f"Action: {row['Recommendation']}"
                )


# ============================================================
# TTC ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "⏱️ Time-to-Collision Analysis"
)


ttc_chart = go.Figure()


ttc_chart.add_trace(
    go.Bar(
        x=df["Vehicle"],
        y=[
            min(
                value,
                30
            )
            for value in df["TTC"]
        ],
        text=df["TTC"],
        textposition="auto"
    )
)


ttc_chart.update_layout(
    height=400,
    yaxis_title="Estimated TTC (seconds)",
    xaxis_title="Vehicle"
)


st.plotly_chart(
    ttc_chart,
    use_container_width=True
)


st.caption(
    "TTC is a simplified demonstration metric using "
    "estimated separation and relative speed. It is "
    "not a certified collision-warning calculation."
)


# ============================================================
# RISK HISTORY
# ============================================================

st.divider()

st.subheader(
    "📈 Historical Risk Monitoring"
)


if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    history_df = history_df.set_index(
        "Step"
    )

    st.line_chart(
        history_df
    )

else:

    st.info(
        "Start the monitoring system to generate "
        "historical risk data."
    )


# ============================================================
# AI EXPLAINABILITY
# ============================================================

st.divider()

st.subheader(
    "🧠 Explainable AI"
)


importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": model.feature_importances_

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


fig_importance = go.Figure()


fig_importance.add_trace(
    go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h"
    )
)


fig_importance.update_layout(
    height=400,
    xaxis_title="Importance",
    yaxis_title="Safety Feature"
)


st.plotly_chart(
    fig_importance,
    use_container_width=True
)


st.write(
    "MineVision considers visibility, speed, "
    "vehicle separation, traffic density, "
    "restricted zones and TTC when generating "
    "its predictive safety classification."
)


# ============================================================
# COMPUTER VISION
# ============================================================

st.divider()

st.subheader(
    "📷 Computer Vision"
)


def run_yolo(image):

    if not YOLO_AVAILABLE:

        return None, 0, []


    try:

        detector = YOLO(
            "yolo11n.pt"
        )

        results = detector.predict(
            source=np.array(image),
            conf=0.35,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        vehicle_classes = {
            "car",
            "truck",
            "bus"
        }

        detections = []


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = detector.names[
                class_id
            ]


            if class_name in vehicle_classes:

                detections.append({

                    "Object": class_name,

                    "Confidence": round(
                        confidence * 100,
                        1
                    )

                })


        return (
            annotated,
            len(detections),
            detections
        )


    except Exception:

        return None, 0, []


if vision_mode == "Upload Image" and uploaded_file:

    input_image = Image.open(
        uploaded_file
    ).convert("RGB")


    if YOLO_AVAILABLE:

        with st.spinner(
            "Running computer vision..."
        ):

            annotated, count, detections = run_yolo(
                input_image
            )


        if annotated is not None:

            c1, c2 = st.columns(2)


            with c1:

                st.image(
                    input_image,
                    caption="Camera Input",
                    use_container_width=True
                )


            with c2:

                st.image(
                    annotated,
                    caption="MineVision Detection",
                    use_container_width=True
                )


            st.metric(
                "Detected Vehicles",
                count
            )


            if detections:

                st.dataframe(
                    pd.DataFrame(detections),
                    use_container_width=True,
                    hide_index=True
                )


        else:

            st.warning(
                "The vision model could not process "
                "this image."
            )

    else:

        st.warning(
            "YOLO is not available in the current "
            "environment."
        )


elif vision_mode == "Camera":

    camera_image = st.camera_input(
        "Capture mine-road image"
    )


    if camera_image:

        input_image = Image.open(
            camera_image
        ).convert("RGB")


        if YOLO_AVAILABLE:

            with st.spinner(
                "Analyzing captured image..."
            ):

                annotated, count, detections = run_yolo(
                    input_image
                )


            if annotated is not None:

                c1, c2 = st.columns(2)


                with c1:

                    st.image(
                        input_image,
                        caption="Camera",
                        use_container_width=True
                    )


                with c2:

                    st.image(
                        annotated,
                        caption="AI Vehicle Detection",
                        use_container_width=True
                    )


                st.metric(
                    "Vehicles Detected",
                    count
                )


        else:

            st.warning(
                "YOLO is unavailable."
            )


else:

    st.info(
        "Select Upload Image or Camera from "
        "the sidebar to demonstrate computer vision."
    )


# ============================================================
# CSV REPORT
# ============================================================

st.divider()

st.subheader(
    "📥 Safety Report"
)


report_df = df[
    [
        "Vehicle",
        "Speed",
        "Distance",
        "Required Distance",
        "TTC",
        "Risk",
        "Prediction",
        "Confidence",
        "Reason",
        "Recommendation"
    ]
]


csv_data = report_df.to_csv(
    index=False
)


st.download_button(
    label="📥 Download Safety Report",
    data=csv_data,
    file_name="minevision_v10_safety_report.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🔧 MineVision System Information"
)


info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        "ML Training Samples",
        len(training_data)
    )


with info2:

    st.metric(
        "AI Features",
        len(FEATURES)
    )


with info3:

    st.metric(
        "Model Accuracy",
        f"{model_accuracy * 100:.1f}%"
    )


with info4:

    st.metric(
        "Monitoring Step",
        st.session_state.step
    )


# ============================================================
# ARCHITECTURE
# ============================================================

with st.expander(
    "🏗️ MineVision V10 Architecture"
):

    st.code(
        """
                MINEVISION V10
                       |
        +--------------+--------------+
        |                             |
        v                             v
   COMPUTER VISION               ENVIRONMENT
        |                             |
       YOLO                    Visibility / Fog
        |                     Speed Conditions
        v                     Traffic Level
 Vehicle Detection                  |
        |                             |
        +--------------+--------------+
                       |
                       v
                VEHICLE ANALYSIS
                       |
             +---------+---------+
             |                   |
             v                   v
        Separation              TTC
             |                   |
             +---------+---------+
                       |
                       v
               RANDOM FOREST
                       |
             +---------+---------+
             |         |         |
             v         v         v
           SAFE     CAUTION    DANGER
             |         |         |
             +---------+---------+
                       |
                       v
                ALERT CENTER
                       |
                       v
               SAFETY REPORT
        """,
        language="text"
    )


# ============================================================
# FUTURE DEVELOPMENT
# ============================================================

with st.expander(
    "🚀 Production Development Roadmap"
):

    st.markdown(
        """
        ### V11+

        **1. Mine-specific computer vision**

        Train the detector using real iron-ore mine
        camera footage.

        **2. Real telemetry**

        Integrate vehicle GPS, speed and distance sensors.

        **3. Sensor fusion**

        Combine camera, GPS, radar/LiDAR and visibility
        sensors.

        **4. Mine geofencing**

        Connect the system to actual haul-road maps
        and restricted areas.

        **5. Historical incident analysis**

        Train predictive models using real mine
        safety-event data.

        **6. Operator notification**

        Integrate appropriate alerting mechanisms.

        **7. Validation**

        Perform extensive field testing and safety
        validation before any operational use.
        """
    )


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander(
    "⚠️ Prototype Limitations"
):

    st.warning(
        """
        MineVision V10 is a hackathon proof-of-concept.

        The predictive model currently uses synthetic
        training data.

        The computer-vision component uses a general-purpose
        pretrained object-detection model.

        Vehicle distance and TTC values in the simulation
        are estimates.

        The system does not directly control vehicles.

        MineVision must not be treated as a certified
        collision-avoidance, autonomous-driving or
        operational safety-control system.

        A production implementation would require
        mine-specific datasets, calibrated sensors,
        validation, testing, safety engineering,
        human factors analysis and applicable certification.
        """
    )


# ============================================================
# HACKATHON MODE
# ============================================================

st.divider()

st.subheader(
    "🏆 Hackathon Demonstration"
)


st.markdown(
    """
    ### MineVision's core idea

    **Detect → Analyze → Predict → Alert**

    MineVision combines environmental conditions and
    vehicle information to provide an AI-assisted view
    of potential safety risks during low-visibility
    conditions.

    ### Demonstration sequence

    **1. Start with clear visibility**

    Show the normal mine environment.

    **2. Reduce visibility**

    Move the visibility slider toward 30–60 m.

    **3. Start monitoring**

    Demonstrate changing vehicle risk classifications.

    **4. Show TTC**

    Explain how vehicle separation and relative speed
    are incorporated.

    **5. Upload a vehicle image**

    Demonstrate computer vision.

    **6. Show Explainable AI**

    Explain which features influence the model.

    **7. Download the report**

    Demonstrate that the system can generate a
    structured safety report.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MineVision V10 | AI-Assisted Mine Vehicle Safety"
)

st.caption(
    "Hackathon prototype — not for operational deployment."
)


# ============================================================
# AUTO REFRESH
# ============================================================

if st.session_state.running:

    st.rerun()
