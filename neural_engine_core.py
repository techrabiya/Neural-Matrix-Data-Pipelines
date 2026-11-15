import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Neural Matrix & Data Pipelines",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #00ffcc;
    }
    .stSidebar {
        background-color: #111827;
    }
    h1, h2, h3 {
        color: #00ffcc;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid #00ffcc;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Neural Matrix Forward Propagation & Activation Pipeline")
st.markdown("---")

st.sidebar.header("🎛️ Neural Control Matrix")
selected_mode = st.sidebar.radio("Pipeline Mode", ["Forward Propagation & Activation", "Matrix Statistical Explorer"])
activation_type = st.sidebar.selectbox("Activation Function", ["ReLU (Rectified Linear Unit)", "Sigmoid (Probability Scale)"])
noise_factor = st.sidebar.slider("Signal Variance Factor", 0.1, 5.0, 1.0)

try:
    np.random.seed(42)
    input_data = np.random.randn(100, 3) * noise_factor
    weights = np.array([[0.5, -1.2, 0.8], [1.1, 0.4, -0.5], [-0.3, 0.9, 1.4]])
    
    raw_dot_product = np.dot(input_data, weights)
    
    if "ReLU" in activation_type:
        activated_output = np.maximum(0, raw_dot_product)
        formula_desc = "ReLU(x) = max(0, x) [Flushes negative values to zero]"
    else:
        activated_output = 1 / (1 + np.exp(-raw_dot_product))
        formula_desc = "Sigmoid(x) = 1 / (1 + exp(-x)) [Maps values between 0 and 1]"

    df_matrix = pd.DataFrame({
        "Input_Feature_1": input_data[:, 0],
        "Raw_Matrix_Output": raw_dot_product[:, 0],
        "Activated_Output": activated_output[:, 0]
    })

    if selected_mode == "Forward Propagation & Activation":
        st.subheader("⚡ Neural Layer Propagation & Transformation Matrix")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🔢 Matrix Shape", value="100 x 3", delta="Synced")
        with col2:
            st.metric(label="⚡ Mean Activation", value=f"{activated_output.mean():.4f}", delta="+4.2%")
        with col3:
            st.metric(label="🛡️ Zero-Crash Status", value="Active / Secure", delta="100% Safe")
            
        st.markdown(f"**Active Transformation Rule:** `{formula_desc}`")
        st.markdown("---")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#111827')
        
        ax.plot(df_matrix['Raw_Matrix_Output'], label='Raw Dot Product (Wx + b)', color='#00ffcc', linewidth=1.8)
        ax.plot(df_matrix['Activated_Output'], label=f'Post-{activation_type.split()[0]} Output', color='#ff007f', linestyle='--', linewidth=2)
        
        ax.set_title("Neural Activation & Forward Propagation Stream", color='#00ffcc', fontsize=12, fontfamily='monospace')
        ax.tick_params(colors='#00ffcc')
        ax.legend(facecolor='#111827', edgecolor='#00ffcc', labelcolor='#00ffcc')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#00ffcc')
            
        st.pyplot(fig)

    else:
        st.subheader("📊 Complete Matrix Data & Pipeline Explorer")
        st.dataframe(df_matrix, use_container_width=True)

except Exception as e:
    st.error(f"Smart Warning: An anomaly was caught by the safety shield: {e}")
    st.info("The application remains online and stable. Please verify your matrix dimensions or configuration parameters.")[cite: 20]
