import streamlit as st
import pandas as pd
import pickle

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="ECU Security System",
    layout="wide",
    page_icon="🛡️"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        h1 {color: #2a3f5f;}
        h2 {color: #2a3f5f; border-bottom: 2px solid #2a3f5f;}
        .st-bw { padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL AND DATA
@st.cache_resource
def load_model():
    with open("knn_model.pkl", 'rb') as file:
        knn_model, preprocessor = pickle.load(file)
    return knn_model, preprocessor

@st.cache_data
def load_data():
    return pd.read_csv("ecu_cluster.csv")

# 4. DATA PREPROCESSING
def preprocess_data(df, preprocessor):
    drop_columns = ["ECU_ID", "Operational_State"]
    X = preprocessor.transform(df.drop(columns=drop_columns))
    return X

# 5. ALTERNATIVE ECU SUGGESTION
def suggest_alternatives(
    ecu_id_under_attack, 
    top_n=3, 
    same_type=True, 
    same_protocol=True, 
    same_topology=True, 
    only_active=True,
    max_cpu_load=0,
    exclude_list=None
):
    if exclude_list is None:
        exclude_list = []

    df = load_data()
    knn_model, preprocessor = load_model()

    idx = df[df["ECU_ID"] == ecu_id_under_attack].index[0]
    input_vector = preprocess_data(df.iloc[[idx]], preprocessor)
    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=len(df))

    original_type = df.iloc[idx]["ECU_Type"]
    original_protocol = df.iloc[idx]["Protocol"]
    original_topology = df.iloc[idx]["Network_Topology_Level"]

    suggestions = []
    for dist, i in zip(distances[0], indices[0]):
        if i == idx or df.iloc[i]["ECU_ID"] in exclude_list:
            continue

        candidate = df.iloc[i]

        if only_active and candidate["Operational_State"] != "Active":
            continue
        if same_type and candidate["ECU_Type"] != original_type:
            continue
        if same_protocol and candidate["Protocol"] != original_protocol:
            continue
        if same_topology and candidate["Network_Topology_Level"] != original_topology:
            continue
        if int(candidate["CPU_Load"].strip('%')) > max_cpu_load:
            continue

        suggestions.append((candidate["ECU_ID"], dist))
        if len(suggestions) == top_n:
            break

    return suggestions, distances[0]


# 6. SIDEBAR NAVIGATION
with st.sidebar:
    st.title("ECU Security Dashboard")
    page = st.radio("Navigation", ["Exploratory Analysis", "Attack Response"],
                    format_func=lambda x: "📊 "+x if x == "Exploratory Analysis" else "🛡️ "+x)

# 7. PAGE 1 - EXPLORATORY ANALYSIS
if page == "Exploratory Analysis":
    st.header("📊 ECU Dataset Analysis")
    df = load_data()

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with col2:
        st.subheader("🔍 Data Composition")
        type_dist = df['ECU_Type'].value_counts()
        st.bar_chart(type_dist)

    st.subheader("🌐 Network Topology Distribution")
    topo_dist = df['Network_Topology_Level'].value_counts()
    st.bar_chart(topo_dist)

# 8. PAGE 2 - ECU ATTACK RESPONSE
elif page == "Attack Response":
    st.header("🛡️ ECU Attack Response System")

    # Incident Information
    with st.container():
        st.subheader("🚨 Incident Information")
        col1, col2 = st.columns([2, 1])
        with col1:
            df_data = load_data()
            ecu_id_under_attack = st.selectbox(
                "Affected ECU ID",
                options=df_data["ECU_ID"].unique(),
                index=0
            )
        with col2:
            status = df_data[df_data["ECU_ID"] == ecu_id_under_attack]["Operational_State"].values[0]
            st.metric("Current Status", status, help="Operational status of selected ECU")

    # Show attacked ECU details
    with st.expander("📋 Attacked ECU Specifications", expanded=False):
        attacked_ecu = df_data[df_data["ECU_ID"] == ecu_id_under_attack].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("ECU Type", attacked_ecu['ECU_Type'])
        col2.metric("Manufacturer", attacked_ecu['Manufacturer'])
        col3.metric("CPU Load", attacked_ecu["CPU_Load"])

        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU Speed", f"{attacked_ecu['CPU_Speed_MHz']} MHz")
        col2.metric("Memory", f"{attacked_ecu['Memory_MB']} MB")
        col3.metric("Power Consumption", f"{attacked_ecu['Power_Watts']} W")
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Message Frequency", f"{attacked_ecu['Message_Frequency_msgs']}/sec")
        col2.metric("Network Topology Level", attacked_ecu['Network_Topology_Level'])
        col3.metric("Protocol", attacked_ecu['Protocol'])
        

    # Configuration Section
    with st.expander("⚙️ Recommendation Settings", expanded=True):
        

        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Number of recommendations", 1, 10, 3)
            same_type = st.checkbox("Require same ECU type", value=True)
            max_cpu_load = st.slider("Maximum CPU Load (%)", 0, 100, 100)
        with col2:
            only_active = st.checkbox("Only consider active ECUs", value=True)
            min_power = st.number_input("Minimum power requirement (Watts)", min_value=0, value=50)
            attacked_list = st.multiselect(
                "Exclude already attacked ECUs", 
                options=list(df_data["ECU_ID"].unique()), 
                default=[ecu_id_under_attack]
            )

    # Generate Recommendations
    if st.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Analyzing ECU network..."):
            suggestions, all_distances = suggest_alternatives(
                ecu_id_under_attack, 
                top_n, 
                same_type=same_type, 
                same_protocol=True,
                same_topology=True,
                only_active=only_active,
                max_cpu_load=max_cpu_load,
                exclude_list=attacked_list
            )

            if suggestions:
                st.success("✅ Recommendations generated successfully!")
                st.subheader("💡 Top Replacement Candidates")

                # Get all distances for normalization
                min_dist = min(all_distances)
                max_dist = max(all_distances)

                for idx, (candidate_id, dist) in enumerate(suggestions):
                    ecu_data = df_data.query(f"ECU_ID == '{candidate_id}'").iloc[0]

                    with st.container():
                        cols = st.columns([1, 5, 1])
                        cols[0].metric("Rank", f"#{idx+1}")
                        cols[1].markdown(f"""
                            **ECU ID**: `{candidate_id}`  
                            **Type**: {ecu_data['ECU_Type']}  
                            **Manufacturer**: {ecu_data['Manufacturer']}  
                            **Software Version**: {ecu_data['Software_Version']}
                            **Protocol**: {ecu_data['Protocol']}
                        """)
                        
                        # Calculate normalized similarity score
                        similarity_score = 1 - ((dist - min_dist) / (max_dist - min_dist)) if max_dist != min_dist else 1.0
                        cols[2].metric(
                            "Similarity", 
                            f"{similarity_score:.1%}",
                            help="Relative compatibility score (0% = least similar in selection, 100% = most similar)"
                        )

                        with st.expander("📊 Detailed Specifications"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("CPU Speed", f"{ecu_data['CPU_Speed_MHz']} MHz")
                            col2.metric("Memory", f"{ecu_data['Memory_MB']} MB")
                            col3.metric("Power", f"{ecu_data['Power_Watts']} W")

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Error Rate", f"{ecu_data['Error_Rate_percent']}%")
                            col2.metric("Response Time", f"{ecu_data['Response_Time_ms']} ms")
                            col3.metric("CPU Load", ecu_data["CPU_Load"])

                        st.markdown("---")

                # Summary Table
                st.subheader("📌 Recommendation Summary")
                df_rec = pd.DataFrame(suggestions, columns=["ECU_ID", "Distance"])
                st.dataframe(df_rec, use_container_width=True)
            else:
                st.error("⚠️ No suitable alternatives found with current filters")

    # System Overview
    with st.container():
        st.subheader("🌐 Network Health Overview")
        df = load_data()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total ECUs", len(df))
        active_ecus = df[df["Operational_State"] == "Active"].shape[0]
        col2.metric("Active ECUs", f"{active_ecus} ({active_ecus/len(df):.1%})")
        col3.metric("Average Response Time", f"{df['Response_Time_ms'].mean():.1f} ms")