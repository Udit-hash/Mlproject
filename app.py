import streamlit as st
import pandas as pd
import joblib
import networkx as nx
import osmnx as ox
import folium
import random

from streamlit_folium import st_folium


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Traffic Prediction & Routing",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Traffic Prediction and Smart Routing")

st.write(
    "Enter two locations to compare the shortest route "
    "with a simple traffic-aware route."
)


# -------------------------------------------------
# LOAD DATA AND MODEL
# -------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/traffic_with_clusters.csv"
    )

    return df


@st.cache_resource
def load_model():

    model = joblib.load(
        "data/traffic_random_forest.pkl"
    )

    return model


df = load_data()
model = load_model()


# -------------------------------------------------
# TRAFFIC LEVEL → CLUSTER NUMBER
# -------------------------------------------------

traffic_mapping = (
    df[["traffic_level", "cluster"]]
    .drop_duplicates()
    .set_index("traffic_level")["cluster"]
    .to_dict()
)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "route_result" not in st.session_state:
    st.session_state.route_result = None


# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

st.subheader("📍 Route Details")

col1, col2 = st.columns(2)


with col1:

    source_text = st.text_input(
        "Starting Location",
        placeholder="Example: Indiranagar, Bengaluru"
    )


with col2:

    destination_text = st.text_input(
        "Destination",
        placeholder="Example: Koramangala, Bengaluru"
    )


col3, col4, col5 = st.columns(3)


with col3:

    day_name = st.selectbox(
        "Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


with col4:

    hour = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=18
    )


with col5:

    traffic_level = st.selectbox(
        "Traffic Condition",
        [
            "Low Traffic",
            "Moderate Traffic",
            "High Traffic"
        ]
    )


# -------------------------------------------------
# DAY CONVERSION
# -------------------------------------------------

day_numbers = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

day_of_week = day_numbers[day_name]


# -------------------------------------------------
# ROUTE BUTTON
# -------------------------------------------------

if st.button(
    "🚗 Find Route",
    type="primary"
):

    if not source_text or not destination_text:

        st.warning(
            "Please enter both starting and destination locations."
        )

    else:

        # -------------------------------------------------
        # GEOCODE LOCATIONS
        # -------------------------------------------------

        try:

            source_point = ox.geocode(
                source_text
            )

            destination_point = ox.geocode(
                destination_text
            )

        except Exception:

            st.error(
                "Could not find one of the locations. "
                "Try adding 'Bengaluru, India' to the location name."
            )

            st.stop()


        source_lat, source_lon = source_point
        destination_lat, destination_lon = destination_point


        # -------------------------------------------------
        # DISPLAY LOCATIONS
        # -------------------------------------------------

        st.success(
            "Both locations found successfully!"
        )

        location_col1, location_col2 = st.columns(2)


        with location_col1:

            st.write(
                f"**Starting:** {source_text}"
            )

            st.write(
                f"Latitude: {source_lat:.5f}"
            )

            st.write(
                f"Longitude: {source_lon:.5f}"
            )


        with location_col2:

            st.write(
                f"**Destination:** {destination_text}"
            )

            st.write(
                f"Latitude: {destination_lat:.5f}"
            )

            st.write(
                f"Longitude: {destination_lon:.5f}"
            )


        # -------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # -------------------------------------------------

        cluster = traffic_mapping[
            traffic_level
        ]


        input_data = pd.DataFrame({
            "hour": [hour],
            "day_of_week": [day_of_week],
            "cluster": [cluster]
        })


        predicted_vehicles = model.predict(
            input_data
        )[0]


        # -------------------------------------------------
        # CREATE ROAD GRAPH
        # -------------------------------------------------

        with st.spinner(
            "Downloading road network..."
        ):

            try:

                min_lat = min(
                    source_lat,
                    destination_lat
                )

                max_lat = max(
                    source_lat,
                    destination_lat
                )

                min_lon = min(
                    source_lon,
                    destination_lon
                )

                max_lon = max(
                    source_lon,
                    destination_lon
                )


                # Add small margin around locations
                margin = 0.01


                bbox = (
                    min_lon - margin,
                    min_lat - margin,
                    max_lon + margin,
                    max_lat + margin
                )


                G = ox.graph_from_bbox(
                    bbox,
                    network_type="drive"
                )


            except Exception as e:

                st.error(
                    "Could not download the road network."
                )

                st.write(e)

                st.stop()


        # -------------------------------------------------
        # FIND NEAREST ROAD NODES
        # -------------------------------------------------

        source_node = ox.distance.nearest_nodes(
            G,
            X=source_lon,
            Y=source_lat
        )


        destination_node = ox.distance.nearest_nodes(
            G,
            X=destination_lon,
            Y=destination_lat
        )


        # -------------------------------------------------
        # NORMAL SHORTEST ROUTE
        # -------------------------------------------------

        try:

            normal_route = nx.shortest_path(
                G,
                source=source_node,
                target=destination_node,
                weight="length"
            )

        except nx.NetworkXNoPath:

            st.error(
                "No route could be found between these locations."
            )

            st.stop()


        normal_distance = nx.path_weight(
            G,
            normal_route,
            weight="length"
        )


        # -------------------------------------------------
        # TRAFFIC WEIGHTS
        # -------------------------------------------------

        traffic_factors = {
            "Low Traffic": 1.0,
            "Moderate Traffic": 1.5,
            "High Traffic": 2.0
        }


        traffic_factor = traffic_factors[
            traffic_level
        ]


        # -------------------------------------------------
        # CREATE SIMULATED ROAD CONGESTION
        # -------------------------------------------------

        # This creates different congestion levels
        # on different road segments.
        #
        # This is only for assignment-level demonstration.
        # It is NOT actual road-segment traffic data.

        random.seed(42)


        for u, v, key, data in G.edges(
            keys=True,
            data=True
        ):

            length = data.get(
                "length",
                0
            )

            # Different roads get different traffic conditions
            road_factor = random.choice(
                [1.0, 1.2, 1.5]
            )

            data["traffic_time"] = (
                length
                * traffic_factor
                * road_factor
            )


        # -------------------------------------------------
        # TRAFFIC-AWARE ROUTE
        # -------------------------------------------------

        traffic_route = nx.shortest_path(
            G,
            source=source_node,
            target=destination_node,
            weight="traffic_time"
        )


        traffic_distance = nx.path_weight(
            G,
            traffic_route,
            weight="length"
        )


        traffic_cost = nx.path_weight(
            G,
            traffic_route,
            weight="traffic_time"
        )


        # -------------------------------------------------
        # APPROXIMATE TRAVEL TIME
        # -------------------------------------------------

        # Simple assumed average speed:
        # 30 km/h for assignment-level demonstration

        base_speed_kmh = 30


        normal_time_hours = (
            normal_distance / 1000
        ) / base_speed_kmh


        traffic_time_hours = (
            traffic_cost / 1000
        ) / base_speed_kmh


        normal_time_minutes = (
            normal_time_hours * 60
        )


        traffic_time_minutes = (
            traffic_time_hours * 60
        )


        # -------------------------------------------------
        # CONVERT ROUTES TO COORDINATES
        # -------------------------------------------------

        normal_coordinates = [
            (
                G.nodes[node]["y"],
                G.nodes[node]["x"]
            )
            for node in normal_route
        ]


        traffic_coordinates = [
            (
                G.nodes[node]["y"],
                G.nodes[node]["x"]
            )
            for node in traffic_route
        ]


        # -------------------------------------------------
        # SAVE EVERYTHING IN SESSION STATE
        # -------------------------------------------------

        st.session_state.route_result = {

            "source_text": source_text,

            "destination_text": destination_text,

            "source_lat": source_lat,

            "source_lon": source_lon,

            "destination_lat": destination_lat,

            "destination_lon": destination_lon,

            "day_name": day_name,

            "hour": hour,

            "traffic_level": traffic_level,

            "predicted_vehicles": predicted_vehicles,

            "normal_distance": normal_distance,

            "traffic_distance": traffic_distance,

            "normal_time_minutes": normal_time_minutes,

            "traffic_time_minutes": traffic_time_minutes,

            "normal_coordinates": normal_coordinates,

            "traffic_coordinates": traffic_coordinates
        }


# =================================================
# DISPLAY SAVED RESULTS
# =================================================

if st.session_state.route_result is not None:

    result = st.session_state.route_result


    # -------------------------------------------------
    # DISPLAY PREDICTION
    # -------------------------------------------------

    st.subheader(
        "📊 Traffic Prediction"
    )


    prediction_col1, prediction_col2 = st.columns(2)


    with prediction_col1:

        st.metric(
            "Predicted Vehicles",
            f"{result['predicted_vehicles']:.0f}"
        )


    with prediction_col2:

        st.metric(
            "Traffic Level",
            result["traffic_level"]
        )


    # -------------------------------------------------
    # ROUTE RESULTS
    # -------------------------------------------------

    st.subheader(
        "🛣️ Route Comparison"
    )


    route_col1, route_col2 = st.columns(2)


    with route_col1:

        st.write(
            "### Shortest Distance Route"
        )

        st.metric(
            "Distance",
            f"{result['normal_distance'] / 1000:.2f} km"
        )

        st.metric(
            "Estimated Time",
            f"{result['normal_time_minutes']:.1f} min"
        )


    with route_col2:

        st.write(
            "### Traffic-Aware Route"
        )

        st.metric(
            "Distance",
            f"{result['traffic_distance'] / 1000:.2f} km"
        )

        st.metric(
            "Estimated Time",
            f"{result['traffic_time_minutes']:.1f} min"
        )


    # -------------------------------------------------
    # CREATE FOLIUM MAP
    # -------------------------------------------------

    center_lat = (
        result["source_lat"]
        + result["destination_lat"]
    ) / 2


    center_lon = (
        result["source_lon"]
        + result["destination_lon"]
    ) / 2


    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=13
    )


    # -------------------------------------------------
    # STARTING MARKER
    # -------------------------------------------------

    folium.Marker(
        [
            result["source_lat"],
            result["source_lon"]
        ],
        popup="Starting Location",
        tooltip="Start"
    ).add_to(m)


    # -------------------------------------------------
    # DESTINATION MARKER
    # -------------------------------------------------

    folium.Marker(
        [
            result["destination_lat"],
            result["destination_lon"]
        ],
        popup="Destination",
        tooltip="Destination"
    ).add_to(m)


    # -------------------------------------------------
    # NORMAL ROUTE
    # -------------------------------------------------

    folium.PolyLine(
        result["normal_coordinates"],
        weight=5,
        popup="Shortest Distance Route"
    ).add_to(m)


    # -------------------------------------------------
    # TRAFFIC-AWARE ROUTE
    # -------------------------------------------------

    folium.PolyLine(
        result["traffic_coordinates"],
        weight=5,
        popup="Traffic-Aware Route"
    ).add_to(m)


    # -------------------------------------------------
    # DISPLAY MAP
    # -------------------------------------------------

    st.subheader(
        "🗺️ Route Map"
    )


    st_folium(
        m,
        width=1200,
        height=600,
        returned_objects=[]
    )


    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    st.subheader(
        "📋 Trip Summary"
    )


    summary = pd.DataFrame({

        "Parameter": [

            "Starting Location",

            "Destination",

            "Day",

            "Hour",

            "Traffic Level",

            "Predicted Vehicles",

            "Shortest Route",

            "Traffic-Aware Route"

        ],


        "Value": [

            result["source_text"],

            result["destination_text"],

            result["day_name"],

            f"{result['hour']}:00",

            result["traffic_level"],

            f"{result['predicted_vehicles']:.0f}",

            f"{result['normal_distance'] / 1000:.2f} km",

            f"{result['traffic_distance'] / 1000:.2f} km"

        ]

    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Traffic prediction uses the trained Random Forest model. "
    "Routing uses OpenStreetMap road data and a simple "
    "congestion-adjusted routing weight."
)