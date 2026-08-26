# ============================================================
# MINEVISION V9
# AI + COMPUTER VISION + PREDICTIVE SAFETY
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
import cv2

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
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="MineVision V9",

    page_icon="⛏️",

    layout="wide"

)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 44px;
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<div class="title">⛏️ MineVision V9</div>',

    unsafe_allow_html=True

)

st.markdown(

    '<div class="subtitle">'
    'AI + Computer Vision + Predictive Mine Vehicle Safety'
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


if "vision_results" not in st.session_state:

    st.session_state.vision_results = None


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
# TIME-TO-COLLISION
# ============================================================

def calculate_ttc(
    distance,
    speed_a,
    speed_b
):

    relative_speed = abs(

        speed_a - speed_b

    )


    if relative_speed <= 0:

        return float("inf")


    ttc = distance / (

        relative_speed / 3.6

    )


    return round(
        ttc,
        2
    )


# ============================================================
# TRAINING DATA
# ============================================================

def generate_training_data(
    samples=6000
):

    rows = []


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

        ttc = random.uniform(
            1,
            30
        )


        speed_limit = calculate_speed_limit(
            visibility
        )


        safe_distance = calculate_safe_distance(

            speed,

            visibility

        )


        risk = 0


        # ----------------------------------------------------
        # Visibility
        # ----------------------------------------------------

        if visibility < 30:

            risk += 30

        elif visibility < 60:

            risk += 22

        elif visibility < 100:

            risk += 12

        elif visibility < 150:

            risk += 5


        # ----------------------------------------------------
        # Speed
        # ----------------------------------------------------

        if speed > speed_limit * 1.5:

            risk += 30

        elif speed > speed_limit:

            risk += 22

        elif speed > speed_limit * 0.8:

            risk += 8


        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        if distance < 20:

            risk += 30

        elif distance < 30:

            risk += 25

        elif distance < safe_distance:

            risk += 18

        elif distance < safe_distance * 1.3:

            risk += 8


        # ----------------------------------------------------
        # TTC
        # ----------------------------------------------------

        if ttc < 3:

            risk += 35

        elif ttc < 5:

            risk += 25

        elif ttc < 8:

            risk += 15

        elif ttc < 12:

            risk += 5


        # ----------------------------------------------------
        # Traffic
        # ----------------------------------------------------

        if traffic >= 9:

            risk += 10

        elif traffic >= 6:

            risk += 5


        # ----------------------------------------------------
        # Restricted zone
        # ----------------------------------------------------

        if restricted:

            risk += 20


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
# TRAIN RANDOM FOREST
# ============================================================

@st.cache_resource
def train_model():

    data = generate_training_data(
        6000
    )


    features = [

        "visibility",

        "speed",

        "distance",

        "traffic",

        "restricted",

        "ttc"

    ]


    X = data[
        features
    ]

    y = data[
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

        n_estimators=180,

        max_depth=12,

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

        data,

        accuracy

    )


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
# YOLO MODEL
# ============================================================

@st.cache_resource
def load_yolo():

    if not YOLO_AVAILABLE:

        return None


    try:

        model = YOLO(
            "yolo11n.pt"
        )

        return model

    except Exception:

        return None


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo_detection(
    image
):

    yolo_model = load_yolo()


    if yolo_model is None:

        return (
            None,
            0,
            "YOLO model unavailable"
        )


    image_array = np.array(
        image
    )


    results = yolo_model.predict(

        source=image_array,

        conf=0.35,

        verbose=False

    )


    result = results[0]


    annotated = result.plot()


    detected = 0


    detections = []


    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )


            class_name = (
                yolo_model.names[
                    cls_id
                ]
            )


            # COCO vehicle classes

            vehicle_classes = {

                "car",
                "truck",
                "bus"

            }


            if class_name in vehicle_classes:

                detected += 1


                coordinates = (

                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()

                )


                detections.append({

                    "type": class_name,

                    "confidence": round(

                        confidence * 100,

                        1

                    ),

                    "box": coordinates

                })


    annotated_rgb = cv2.cvtColor(

        annotated,

        cv2.COLOR_BGR2RGB

    )


    return (

        annotated_rgb,

        detected,

        detections

    )


# ============================================================
# SIMULATED VEHICLES
# ============================================================

def create_vehicle(number):

    points = [

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


    point = points[

        (number - 1)
        %
        len(points)

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


# ------------------------------------------------------------

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
    "Environment"
)


visibility = st.sidebar.slider(

    "🌫️ Visibility",

    20,

    200,

    120,

    5

)


vehicle_count = st.sidebar.slider(

    "🚛 Simulated Vehicles",

    3,

    10,

    6

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

    st.session_state.running = False

    st.session_state.step = 0

    st.session_state.initialized = False

    st.session_state.risk_history = []


# ============================================================
# CAMERA
# ============================================================

st.sidebar.divider()


st.sidebar.subheader(
    "📷 Computer Vision"
)


vision_mode = st.sidebar.selectbox(

    "Vision Mode",

    [

        "Disabled",

        "YOLO Image Detection"

    ]

)


uploaded_image = None


if vision_mode == "YOLO Image Detection":

    uploaded_image = st.sidebar.file_uploader(

        "Upload mine-road image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )


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
# INITIALIZE SIMULATION
# ============================================================

if (

    not st.session_state.initialized

    or

    len(

        st.session_state.vehicles

    ) != vehicle_count

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
# VEHICLE ANALYSIS
# ============================================================

vehicle_data = []


for vehicle in st.session_state.vehicles:

    nearest_distance = 999

    nearest_vehicle = "None"

    nearest_speed = 0


    for other in st.session_state.vehicles:

        if vehicle["id"] == other["id"]:

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

            nearest_speed = other["speed"]


    # Map units → approximate demonstration distance

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


    ttc = calculate_ttc(

        real_distance,

        vehicle["speed"],

        nearest_speed

    )


    # --------------------------------------------------------
    # Prevent infinite values in ML input
    # --------------------------------------------------------

    model_ttc = min(

        ttc,

        60

    )


    model_input = pd.DataFrame([{

        "visibility": visibility,

        "speed": vehicle["speed"],

        "distance": real_distance,

        "traffic": vehicle_count,

        "restricted": int(
            restricted
        ),

        "ttc": model_ttc

    }])


    prediction = model.predict(

        model_input

    )[0]


    probabilities = model.predict_proba(

        model_input

    )[0]


    confidence = round(

        max(probabilities) * 100,

        1

    )


    risk_values = {

        "SAFE": 20,

        "CAUTION": 55,

        "DANGER": 90

    }


    risk = risk_values[

        prediction

    ]


    # --------------------------------------------------------
    # Risk explanation
    # --------------------------------------------------------

    reasons = []


    if visibility < 60:

        reasons.append(
            "low visibility"
        )


    if vehicle["speed"] > speed_limit:

        reasons.append(
            "speed above visibility limit"
        )


    if real_distance < safe_distance:

        reasons.append(
            "insufficient separation"
        )


    if ttc < 5:

        reasons.append(
            "low time-to-collision"
        )


    if restricted:

        reasons.append(
            "restricted zone"
        )


    if not reasons:

        reasons.append(
            "conditions within monitored limits"
        )


    explanation = ", ".join(
        reasons
    )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if prediction == "DANGER":

        recommendation = (

            "STOP / ESCALATE TO OPERATOR"

        )


    elif prediction == "CAUTION":

        recommendation = (

            "REDUCE SPEED / INCREASE DISTANCE"

        )


    else:

        recommendation = (

            "NORMAL MONITORED OPERATION"

        )


    vehicle_data.append({

        "Vehicle": vehicle["id"],

        "Speed": vehicle["speed"],

        "Distance": real_distance,

        "Safe Distance": safe_distance,

        "TTC": ttc,

        "Prediction": prediction,

        "Confidence": confidence,

        "Risk": risk,

        "Reason": explanation,

        "Recommendation": recommendation,

        "Nearest": nearest_vehicle,

        "x": vehicle["x"],

        "y": vehicle["y"]

    })


df = pd.DataFrame(
    vehicle_data
)


# ============================================================
# SAFETY METRICS
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
# SAVE RISK HISTORY
# ============================================================

if st.session_state.step > 0:

    st.session_state.risk_history.append({

        "Step": st.session_state.step,

        "Risk": average_risk,

        "Safety Score": safety_score

    })


if len(
    st.session_state.risk_history
) > 50:

    st.session_state.risk_history = (

        st.session_state.risk_history[-50:]

    )


# ============================================================
# DASHBOARD
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

        "🚨 Danger",

        danger_count

    )


# ============================================================
# COMPUTER VISION
# ============================================================

st.divider()

st.subheader(
    "📷 Computer Vision Analysis"
)


if uploaded_image is not None:

    image = Image.open(

        uploaded_image

    ).convert(
        "RGB"
    )


    with st.spinner(
        "Running YOLO vehicle detection..."
    ):

        result = run_yolo_detection(

            image

        )


    if result[0] is not None:

        annotated_image = result[0]

        detected_count = result[1]

        detections = result[2]


        left, right = st.columns(2)


        with left:

            st.image(

                image,

                caption="Camera Input",

                use_container_width=True

            )


        with right:

            st.image(

                annotated_image,

                caption="MineVision YOLO Detection",

                use_container_width=True

            )


        st.metric(

            "Detected Vehicles",

            detected_count

        )


        if detected_count > 0:

            st.success(

                f"Vision system detected "
                f"{detected_count} vehicle(s)."

            )


            if detections:

                detection_df = pd.DataFrame(

                    [

                        {

                            "Object": d["type"],

                            "Confidence": (

                                f"{d['confidence']}%"

                            )

                        }

                        for d in detections

                    ]

                )


                st.dataframe(

                    detection_df,

                    use_container_width=True,

                    hide_index=True

                )

        else:

            st.warning(

                "No supported vehicle classes "
                "were detected."

            )


    else:

        st.warning(

            "YOLO could not be loaded. "
            "The simulation and ML system can "
            "still operate."

        )


else:

    st.info(

        "Upload a mine-road image from the "
        "sidebar to demonstrate YOLO detection."

    )


# ============================================================
# MINE MAP
# ============================================================

st.divider()

st.subheader(
    "🗺️ Live Mine Vehicle Risk Map"
)


fig = go.Figure()


# Mine boundary

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


# Roads

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


# Vehicle colors

colors = []


for prediction in df[
    "Prediction"
]:

    if prediction == "DANGER":

        colors.append(
            "red"
        )

    elif prediction == "CAUTION":

        colors.append(
            "orange"
        )

    else:

        colors.append(
            "green"
        )


# Vehicles

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

        customdata=df[

            [

                "Prediction",

                "Confidence",

                "Speed",

                "Distance",

                "TTC"

            ]

        ].values,

        hovertemplate=(

            "<b>%{text}</b><br>"

            "Risk: %{customdata[0]}<br>"

            "Confidence: %{customdata[1]}%<br>"

            "Speed: %{customdata[2]} km/h<br>"

            "Distance: %{customdata[3]} m<br>"

            "TTC: %{customdata[4]} s"

            "<extra></extra>"

        )

    )

)


# Restricted area

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


# Excavation

fig.add_shape(

    type="circle",

    x0=10,

    y0=65,

    x1=30,

    y1=85,

    line=dict(
        width=2
    ),

    fillcolor="rgba(100,100,100,0.2)"

)


fig.add_annotation(

    x=20,

    y=75,

    text="⛏️ EXCAVATION",

    showarrow=False

)


# Loading

fig.add_shape(

    type="rect",

    x0=70,

    y0=70,

    x1=90,

    y1=90,

    line=dict(
        width=2
    ),

    fillcolor="rgba(100,100,100,0.2)"

)


fig.add_annotation(

    x=80,

    y=80,

    text="🏭 LOADING",

    showarrow=False

)


# Dump

fig.add_shape(

    type="rect",

    x0=70,

    y0=10,

    x1=92,

    y1=30,

    line=dict(
        width=2
    ),

    fillcolor="rgba(100,100,100,0.2)"

)


fig.add_annotation(

    x=81,

    y=20,

    text="🏗️ DUMP",

    showarrow=False

)


# Fog overlay

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


fig.update_layout(

    height=650,

    xaxis=dict(

        range=[

            0,

            100

        ],

        showgrid=False

    ),

    yaxis=dict(

        range=[

            0,

            100

        ],

        showgrid=False,

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
# AI ALERTS
# ============================================================

st.subheader(
    "🚨 Predictive Safety Alerts"
)


if danger_count > 0:

    st.error(

        f"CRITICAL: {danger_count} vehicle(s) "
        f"are currently classified as DANGER."

    )


    for _, row in df[

        df["Prediction"] == "DANGER"

    ].iterrows():

        st.error(

            f"🚛 {row['Vehicle']} | "

            f"{row['Reason']} | "

            f"TTC: {row['TTC']} s | "

            f"{row['Recommendation']}"

        )


elif caution_count > 0:

    st.warning(

        f"WARNING: {caution_count} vehicle(s) "
        f"require attention."

    )


else:

    st.success(

        "SYSTEM STATUS: All monitored vehicles "
        "are currently in the SAFE class."

    )


# ============================================================
# VEHICLE TABLE
# ============================================================

st.subheader(
    "🚛 Vehicle Intelligence"
)


table = df[

    [

        "Vehicle",

        "Speed",

        "Distance",

        "Safe Distance",

        "TTC",

        "Prediction",

        "Confidence",

        "Risk",

        "Reason",

        "Recommendation"

    ]

].copy()


table.columns = [

    "Vehicle",

    "Speed km/h",

    "Distance m",

    "Required Distance m",

    "TTC seconds",

    "AI Prediction",

    "Confidence",

    "Risk Score",

    "Risk Explanation",

    "Recommendation"

]


st.dataframe(

    table,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# TTC PANEL
# ============================================================

st.subheader(
    "⏱️ Time-to-Collision Analysis"
)


ttc_display = df[

    [

        "Vehicle",

        "Distance",

        "TTC",

        "Prediction"

    ]

].copy()


ttc_display.columns = [

    "Vehicle",

    "Distance (m)",

    "Estimated TTC (s)",

    "AI Risk"

]


st.dataframe(

    ttc_display,

    use_container_width=True,

    hide_index=True

)


st.caption(

    "TTC is a simplified demonstration metric based on "
    "relative speed and estimated separation. It is not "
    "a certified collision-warning calculation."

)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🧠 AI Feature Importance"
)


importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": model.feature_importances_

})


importance = importance.sort_values(

    "Importance",

    ascending=False

)


st.bar_chart(

    importance.set_index(
        "Feature"
    )

)


# ============================================================
# RISK HISTORY
# ============================================================

st.subheader(
    "📈 Risk History"
)


if st.session_state.risk_history:

    history = pd.DataFrame(

        st.session_state.risk_history

    )


    history = history.set_index(
        "Step"
    )


    st.line_chart(
        history
    )

else:

    st.info(

        "Start the simulation to generate "
        "risk history."

    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "🤖 Machine Learning Model"
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(

        "Training Samples",

        len(training_data)

    )


with m2:

    st.metric(

        "Input Features",

        len(FEATURES)

    )


with m3:

    st.metric(

        "Validation Accuracy",

        f"{model_accuracy * 100:.1f}%"

    )


# ============================================================
# TRAINING DATA
# ============================================================

with st.expander(
    "📚 View Training Dataset"
):

    st.write(

        "The current prototype uses synthetic "
        "training data."

    )


    st.dataframe(

        training_data.head(25),

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# ARCHITECTURE
# ============================================================

with st.expander(
    "🏗️ MineVision V9 Architecture"
):

    st.code(
        """
CAMERA
   |
   v
YOLO OBJECT DETECTION
   |
   v
VEHICLE COUNT / LOCATION
   |
   +----------------------+
   |                      |
   v                      v
GPS / SIMULATION      ENVIRONMENT
   |                  |
   |                  +-- Visibility
   |                  +-- Speed
   |                  +-- Traffic
   |                  +-- Zone
   |
   v
DISTANCE ESTIMATION
   |
   v
TIME-TO-COLLISION
   |
   v
RANDOM FOREST
   |
   +----------+----------+
   |          |          |
   v          v          v
 SAFE      CAUTION     DANGER
   |          |          |
   +----------+----------+
              |
              v
       SAFETY RECOMMENDATION
        """,
        language="text"
    )


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander(
    "⚠️ Prototype Limitations"
):

    st.warning(
        """
        MineVision V9 is a hackathon prototype.

        The YOLO component uses a general-purpose pretrained
        object-detection model and is not trained specifically
        for iron-ore mine environments.

        The training dataset for the risk model is synthetic.

        Distance and TTC values in the simulation are estimates.

        This system must not be used as a real-world autonomous
        collision-avoidance or vehicle-control system.

        A production system would require mine-specific
        datasets, calibrated sensors, validation, testing,
        safety engineering and appropriate certification.
        """
    )


# ============================================================
# FUTURE V10
# ============================================================

with st.expander(
    "🚀 V10 Development Roadmap"
):

    st.write(
        """
        Possible V10 upgrades:

        • Real-time camera stream
        • Vehicle tracking IDs
        • Mine-specific YOLO training
        • GPS coordinates
        • Real sensor integration
        • Weather API
        • Visibility sensor
        • Automatic geofencing
        • Historical incident database
        • Operator notification
        • Digital mine twin
        • Multi-camera monitoring
        """
    )


# ============================================================
# STATUS
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

    if YOLO_AVAILABLE:

        st.write(
            "**Vision:** YOLO"
        )

    else:

        st.write(
            "**Vision:** Unavailable"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MineVision V9 | AI + Computer Vision + Predictive Safety"
)

st.caption(
    "Prototype for hackathon demonstration purposes."
)


# ============================================================
# REFRESH
# ============================================================

if st.session_state.running:

    st.rerun()
