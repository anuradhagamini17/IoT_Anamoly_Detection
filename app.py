import os
import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="IoT Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/anomaly_model.pkl"
DATASET_PATH = "dataset/iot_traffic.csv"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

st.title("🛡️ IoT-Based Smart Home Anomaly Detection")

st.markdown(
    """
    ### Machine Learning Assisted Anomaly Detection System

    This application analyzes network traffic features and uses a
    trained Random Forest classifier to identify potentially anomalous
    traffic in a sample IoT smart home network.
    """
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Please run train_model.py first."
    )
    st.stop()

if not os.path.exists(DATASET_PATH):
    st.error(
        "Dataset not found. Please run train_model.py first."
    )
    st.stop()


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

try:
    model = load_model()
    data = load_dataset()

except Exception as error:
    st.error("Unable to load the model or dataset.")
    st.code(str(error))
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "Dashboard",
        "Detect Anomaly",
        "Dataset Explorer",
        "About Project"
    ]
)


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.header("📊 Network Traffic Dashboard")

    total_records = len(data)

    normal_records = int((data["label"] == 0).sum())

    anomaly_records = int((data["label"] == 1).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Traffic Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Normal Records",
        f"{normal_records:,}"
    )

    col3.metric(
        "Anomalous Records",
        f"{anomaly_records:,}"
    )

    st.divider()

    st.subheader("Traffic Classification Distribution")

    counts = data["label"].value_counts().reindex(
        [0, 1],
        fill_value=0
    )

    fig, ax = plt.subplots()

    ax.bar(
        ["Normal Traffic", "Anomalous Traffic"],
        counts.values
    )

    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of Network Traffic")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Network Traffic Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    st.info(
        "The displayed statistics describe the generated sample dataset. "
        "They do not represent live network traffic."
    )


# ==========================================
# DETECT ANOMALY PAGE
# ==========================================

elif page == "Detect Anomaly":

    st.header("🔍 Detect Network Traffic Anomaly")

    st.write(
        "Enter the network traffic details below to obtain a prediction."
    )

    with st.form("anomaly_detection_form"):

        packet_size = st.number_input(
            "Packet Size (bytes)",
            min_value=40,
            max_value=1500,
            value=500,
            step=10
        )

        packet_rate = st.number_input(
            "Packet Rate",
            min_value=1,
            max_value=999,
            value=100,
            step=10
        )

        flow_duration = st.number_input(
            "Flow Duration",
            min_value=1,
            max_value=9999,
            value=1000,
            step=100
        )

        failed_connections = st.number_input(
            "Failed Connections",
            min_value=0,
            max_value=19,
            value=0,
            step=1
        )

        submitted = st.form_submit_button(
            "Analyze Network Traffic"
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                packet_size,
                packet_rate,
                flow_duration,
                failed_connections
            ]],
            columns=[
                "packet_size",
                "packet_rate",
                "flow_duration",
                "failed_connections"
            ]
        )

        prediction = int(model.predict(input_data)[0])

        st.divider()

        st.subheader("Analysis Result")

        if prediction == 1:

            st.error(
                "⚠️ ANOMALOUS TRAFFIC DETECTED"
            )

            st.write(
                "The trained model classified the supplied "
                "traffic features as anomalous."
            )

        else:

            st.success(
                "✅ NORMAL TRAFFIC"
            )

            st.write(
                "The trained model classified the supplied "
                "traffic features as normal."
            )

        st.subheader("Submitted Traffic Features")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        st.caption(
            "This is a model prediction based on synthetic training data. "
            "It is not a confirmed security incident or a validated "
            "real-world attack detector."
        )


# ==========================================
# DATASET EXPLORER PAGE
# ==========================================

elif page == "Dataset Explorer":

    st.header("📁 Dataset Explorer")

    st.write(
        "Explore the generated IoT network traffic dataset."
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric(
        "Number of Rows",
        len(data)
    )

    col2.metric(
        "Number of Columns",
        len(data.columns)
    )

    st.subheader("Complete Dataset")

    st.dataframe(
        data,
        use_container_width=True
    )

    st.subheader("Filter by Traffic Type")

    traffic_type = st.selectbox(
        "Select traffic type:",
        [
            "All Traffic",
            "Normal Traffic",
            "Anomalous Traffic"
        ]
    )

    if traffic_type == "Normal Traffic":

        filtered_data = data[data["label"] == 0]

    elif traffic_type == "Anomalous Traffic":

        filtered_data = data[data["label"] == 1]

    else:

        filtered_data = data

    st.write(
        "Records displayed:",
        len(filtered_data)
    )

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

    csv_data = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Selected Dataset as CSV",
        data=csv_data,
        file_name="iot_traffic_data.csv",
        mime="text/csv"
    )


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.header("ℹ️ About the Project")

    st.subheader("Project Title")

    st.write(
        "Machine Learning Assisted Anomaly Detection "
        "for IoT-Based Smart Home Networks"
    )

    st.subheader("Problem Statement")

    st.write(
        """
        Smart home networks contain connected devices such as smart
        cameras, sensors, and other IoT equipment. Unusual network
        activity may indicate a malfunction or a security threat.
        Manually inspecting network traffic can be difficult.
        """
    )

    st.subheader("Proposed Solution")

    st.write(
        """
        This prototype uses a Random Forest machine learning model
        to classify network traffic based on packet size, packet rate,
        flow duration, and failed connections.
        """
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Random Forest classification
        - Streamlit
        - Matplotlib
        """
    )

    st.subheader("Current Implementation")

    st.write(
        """
        The current prototype generates synthetic network traffic,
        trains a classifier, evaluates its predictions, and provides
        a dashboard for exploring data and testing individual inputs.
        """
    )

    st.warning(
        """
        The current prototype does not capture live IoT network packets.
        Its labels are generated using predefined rules, so its reported
        accuracy should not be interpreted as real-world attack detection
        accuracy.
        """
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "IoT Anomaly Detection Prototype | "
    "Machine Learning-Based Network Traffic Analysis"
)