# ============================================================
# MINEVISION V8
# AI + COMPUTER VISION MINE VEHICLE SAFETY SYSTEM
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MineVision V8",
    page_icon="⛏️",
    layout="wide"
)


# ============================================================
# CSS
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

div[data-testid="stMetric"] {
    background-color: white;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

.alert-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">⛏️ MineVision V8</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI + Computer Vision Safety System for Open-Cast Mine Vehicles'
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
# TRAINING DATA
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

        risk = 0

        # Visibility risk

        if visibility < 30:
            risk += 35

        elif visibility < 60:
            risk += 25

        elif visibility < 100:
            risk += 15

        elif visibility < 150:
            risk += 5


        # Speed risk

        if speed > speed_limit * 1.5:
            risk += 30

        elif speed > speed_limit:
            risk += 25

        elif speed > speed_limit * 0.8:
            risk += 10


        # Distance risk

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


        # Noise

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
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = generate_training_data(
        5000
    )


    features = [

        "visibility",
        "speed",
        "distance",
        "traffic",
        "restricted"

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

        n_estimators=150,

        max_depth=10,

        random_state=42

    )


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
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
    "restricted"

]


# ============================================================
# COMPUTER VISION
# ============================================================

def detect_vehicles(image):

    """
    Basic OpenCV vehicle detection.

    This is a prototype detector using image processing.
    It is NOT a production-grade object detection model.
    """

    image_array = np.array(
        image
    )

    frame = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    scale = 900 / width

    if width > 900:

        frame = cv2.resize(

            frame,

            (
                900,
                int(height * scale)
            )

        )


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY

    )


    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(

        gray,

        (5, 5),

        0

    )


    # --------------------------------------------------------
    # Edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(

        blurred,

        50,

        150

    )


    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        edges,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    detections = []


    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )


        area = w * h


        # Simple filtering

        if (

            area > 1500

            and

            w > 35

            and

            h > 25

            and

            w < frame.shape[1] * 0.8

        ):

            aspect_ratio = w / max(
                h,
                1
            )


            if 0.6 < aspect_ratio < 5:

                detections.append(
                    (x, y, w, h)
                )


    # --------------------------------------------------------
    # Remove overlapping detections
    # --------------------------------------------------------

    final_boxes = []


    for box in detections:

        x, y, w, h = box

        keep = True


        for existing in final_boxes:

            ex, ey, ew, eh = existing


            center_x = x + w / 2

            center_y = y + h / 2


            if (

                ex <= center_x <= ex + ew

                and

                ey <= center_y <= ey + eh

            ):

                keep = False

                break


        if keep:

            final_boxes.append(
                box
            )


    # --------------------------------------------------------
    # Draw results
    # --------------------------------------------------------

    for index, box in enumerate(
        final_boxes
    ):

        x, y, w, h = box


        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            3

        )


        label = (
            f"Vehicle {index + 1}"
        )


        cv2.putText(

            frame,

            label,

            (x, max(y - 10, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2

        )


    # --------------------------------------------------------
    # Convert back to RGB
    # --------------------------------------------------------

    result = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    return result, len(final_boxes)


# ============================================================
# VEHICLE SIMULATION
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
    "Simulation Controls"
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


# ============================================================
# CAMERA INPUT
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📷 Camera Analysis"
)


camera_mode = st.sidebar.radio(

    "Camera Source",

    [

        "Disabled",

        "Upload Image"

    ]

)


uploaded_image = None


if camera_mode == "Upload Image":

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
# INITIALIZE VEHICLES
# ============================================================

if (

    not st.session_state.initialized

    or

    len(
        st.session_state.vehicles
    )
    != vehicle_count

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
    # ML INPUT
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


    prediction = model.predict(
        model_input
    )[0]


    probability = model.predict_proba(
        model_input
    )[0]


    confidence = round(

        max(probability)
        *
        100,

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

        "Risk": risk,

        "Recommendation": recommendation,

        "Nearest Vehicle": nearest_vehicle,

        "x": vehicle["x"],

        "y": vehicle["y"]

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
# TOP METRICS
# ============================================================

st.subheader(
    "🤖 MineVision Predictive Dashboard"
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

        "🔴 High Risk",

        danger_count

    )


st.divider()


# ============================================================
# COMPUTER VISION PANEL
# ============================================================

st.subheader(
    "📷 Computer Vision Vehicle Detection"
)


if uploaded_image is not None:

    image = Image.open(
        uploaded_image
    )


    result_image, detected_count = detect_vehicles(
        image
    )


    left, right = st.columns(2)


    with left:

        st.image(

            image,

            caption="Original Camera Image",

            use_container_width=True

        )


    with right:

        st.image(

            result_image,

            caption=(
                "MineVision Detection Result"
            ),

            use_container_width=True

        )


    if detected_count > 0:

        st.success(

            f"Computer Vision detected "
            f"{detected_count} possible vehicle/object "
            f"region(s)."

        )

    else:

        st.warning(

            "No likely vehicle regions detected "
            "by the prototype vision detector."

        )


else:

    st.info(

        "Upload a mine-road image from the sidebar "
        "to demonstrate computer-vision analysis."

    )


# ============================================================
# MINE MAP
# ============================================================

st.subheader(
    "🗺️ Predictive Mine Operations Map"
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


# Excavation area

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


# Loading area

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


# Dump area

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


# Restricted zone

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
# AI ALERTS
# ============================================================

st.subheader(
    "🚨 AI Safety Alerts"
)


if danger_count > 0:

    st.error(

        f"AI ALERT: {danger_count} vehicle(s) "
        f"currently classified as HIGH RISK."

    )


    for _, row in df[
        df["Prediction"] == "DANGER"
    ].iterrows():

        st.error(

            f"🚛 {row['Vehicle']} | "
            f"DANGER | "
            f"Confidence {row['Confidence']}% | "
            f"{row['Recommendation']}"

        )


elif caution_count > 0:

    st.warning(

        f"AI WARNING: {caution_count} vehicle(s) "
        f"require additional attention."

    )


else:

    st.success(

        "AI STATUS: All monitored vehicles "
        "are currently within the SAFE class."

    )


# ============================================================
# VEHICLE TABLE
# ============================================================

st.subheader(
    "🚛 Vehicle Intelligence"
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

    "Risk",

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
# RISK HISTORY
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

        "Model Accuracy",

        f"{model_accuracy * 100:.1f}%"

    )


# ============================================================
# EXPLAINABILITY
# ============================================================

with st.expander(
    "🔍 How MineVision V8 Works"
):

    st.write(
        """
        MineVision V8 combines two prototype systems.

        1. COMPUTER VISION

        OpenCV processes an uploaded mine-road image
        and identifies regions that may correspond
        to vehicles or large objects.

        2. MACHINE LEARNING

        A Random Forest model receives:

        • Visibility
        • Vehicle speed
        • Vehicle separation
        • Traffic density
        • Restricted-zone status

        and predicts:

        SAFE
        CAUTION
        DANGER

        3. DECISION SUPPORT

        The prediction is converted into an operational
        recommendation for the mine operator.
        """
    )


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

with st.expander(
    "⚠️ Prototype Limitations"
):

    st.warning(
        """
        This is a hackathon prototype.

        The computer-vision component uses basic OpenCV
        image processing and should NOT be treated as a
        certified vehicle-detection or collision-avoidance
        system.

        The machine-learning model is trained on synthetic
        data for demonstration.

        A real deployment would require validated mine
        datasets, calibrated sensors, robust object
        detection,
        field testing, safety validation and appropriate
        engineering controls.
        """
    )


# ============================================================
# FUTURE DEVELOPMENT
# ============================================================

with st.expander(
    "🚀 V9 Development Path"
):

    st.write(
        """
        Future MineVision versions could include:

        • YOLO-based vehicle detection
        • Real-time video processing
        • GPS integration
        • IoT visibility sensors
        • Weather data
        • Automatic vehicle tracking
        • Time-to-collision estimation
        • Historical incident prediction
        • Mine digital twin
        • Operator notification system
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
        "**Vision:** OpenCV"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MineVision V8 | AI + Computer Vision Mine Safety Prototype"
)

st.caption(
    "All simulated operational data are for demonstration purposes."
)


# ============================================================
# REFRESH
# ============================================================

if st.session_state.running:

    st.rerun()
