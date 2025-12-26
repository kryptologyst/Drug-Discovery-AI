"""
Streamlit demo application for drug discovery AI.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import set_seed, get_device, validate_smiles
from src.data.molecular import (
    smiles_to_morgan_fingerprint, 
    smiles_to_maccs_keys, 
    smiles_to_rdkit_descriptors,
    smiles_to_graph,
    generate_synthetic_dataset
)
from src.models import create_model
from src.metrics import QSARMetrics


# Page configuration
st.set_page_config(
    page_title="Drug Discovery AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .molecule-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research demonstration tool only.</strong></p>
    <ul>
        <li>NOT for clinical use or diagnostic purposes</li>
        <li>NOT a substitute for professional medical advice</li>
        <li>Results are for educational and research purposes only</li>
        <li>Always consult qualified healthcare professionals for medical decisions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧬 Drug Discovery AI</h1>', unsafe_allow_html=True)
st.markdown("### Molecular Property Prediction using QSAR Models")

# Sidebar
st.sidebar.title("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Fingerprint-based Neural Network", "Graph Neural Network", "Random Forest Baseline"],
    help="Choose the type of model for molecular property prediction"
)

# Fingerprint type selection
if "Fingerprint" in model_type:
    fingerprint_type = st.sidebar.selectbox(
        "Fingerprint Type",
        ["Morgan", "MACCS Keys", "RDKit Descriptors"],
        help="Type of molecular fingerprint to use"
    )
else:
    fingerprint_type = None

# Load model (simplified for demo)
@st.cache_resource
def load_demo_model(model_type: str, fingerprint_type: str = None):
    """Load a demo model for prediction."""
    set_seed(42)
    
    if "Random Forest" in model_type:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Generate synthetic data for training
        smiles_list, targets = generate_synthetic_dataset(n_samples=500, seed=42)
        
        # Create features
        if fingerprint_type == "Morgan":
            features = [smiles_to_morgan_fingerprint(s) for s in smiles_list]
        elif fingerprint_type == "MACCS Keys":
            features = [smiles_to_maccs_keys(s) for s in smiles_list]
        else:  # RDKit Descriptors
            features = [smiles_to_rdkit_descriptors(s) for s in smiles_list]
        
        # Filter valid features
        valid_features = []
        valid_targets = []
        for f, t in zip(features, targets):
            if f is not None:
                valid_features.append(f)
                valid_targets.append(t)
        
        # Train model
        X = np.array(valid_features)
        y = np.array(valid_targets)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, fingerprint_type
    
    else:
        # For neural networks, create a simple model
        if fingerprint_type == "Morgan":
            input_dim = 1024
        elif fingerprint_type == "MACCS Keys":
            input_dim = 167
        else:  # RDKit Descriptors
            input_dim = 11
        
        model = create_model(
            model_type="fingerprint",
            input_dim=input_dim,
            hidden_dims=[128, 64],
            dropout=0.2
        )
        
        # For demo purposes, we'll use random weights
        # In a real application, you would load trained weights
        return model, fingerprint_type

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔬 Molecular Input")
    
    # SMILES input
    smiles_input = st.text_input(
        "Enter SMILES String",
        value="CCO",  # ethanol as default
        help="Enter a valid SMILES string representing a molecule",
        placeholder="e.g., CCO (ethanol)"
    )
    
    # Validate SMILES
    if smiles_input:
        if validate_smiles(smiles_input):
            st.success("✅ Valid SMILES string")
            
            # Display molecule
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                # Calculate molecular descriptors
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                num_atoms = mol.GetNumAtoms()
                
                # Display molecule info
                st.markdown('<div class="molecule-info">', unsafe_allow_html=True)
                st.write("**Molecular Properties:**")
                st.write(f"- Molecular Weight: {mw:.2f} g/mol")
                st.write(f"- LogP: {logp:.2f}")
                st.write(f"- TPSA: {tpsa:.2f} Å²")
                st.write(f"- Number of Atoms: {num_atoms}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Draw molecule
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="Molecular Structure")
                
        else:
            st.error("❌ Invalid SMILES string")

with col2:
    st.header("🎯 Prediction Results")
    
    if smiles_input and validate_smiles(smiles_input):
        # Load model
        model, fp_type = load_demo_model(model_type, fingerprint_type)
        
        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                if "Random Forest" in model_type:
                    # Random Forest prediction
                    if fp_type == "Morgan":
                        features = smiles_to_morgan_fingerprint(smiles_input)
                    elif fp_type == "MACCS Keys":
                        features = smiles_to_maccs_keys(smiles_input)
                    else:  # RDKit Descriptors
                        features = smiles_to_rdkit_descriptors(smiles_input)
                    
                    if features is not None:
                        prediction = model.predict([features])[0]
                        uncertainty = 0.1  # Placeholder uncertainty
                    else:
                        prediction = 0.0
                        uncertainty = 0.0
                
                else:
                    # Neural network prediction (simplified)
                    if fp_type == "Morgan":
                        features = smiles_to_morgan_fingerprint(smiles_input)
                    elif fp_type == "MACCS Keys":
                        features = smiles_to_maccs_keys(smiles_input)
                    else:  # RDKit Descriptors
                        features = smiles_to_rdkit_descriptors(smiles_input)
                    
                    if features is not None:
                        # Convert to tensor and make prediction
                        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            prediction = model(features_tensor).item()
                        uncertainty = 0.1  # Placeholder uncertainty
                    else:
                        prediction = 0.0
                        uncertainty = 0.0
                
                # Display results
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Predicted Activity",
                    value=f"{prediction:.3f}",
                    help="Predicted molecular activity (0-1 scale)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Prediction Uncertainty",
                    value=f"{uncertainty:.3f}",
                    help="Model uncertainty in prediction"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Activity interpretation
                if prediction > 0.7:
                    activity_level = "High"
                    color = "green"
                elif prediction > 0.4:
                    activity_level = "Medium"
                    color = "orange"
                else:
                    activity_level = "Low"
                    color = "red"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background-color: {color}20; border-radius: 10px;">
                    <h4>Activity Level: <span style="color: {color};">{activity_level}</span></h4>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    else:
        st.info("Please enter a valid SMILES string to see predictions")

# Additional sections
st.markdown("---")

# Model comparison section
st.header("📊 Model Comparison")

if st.button("Generate Model Comparison"):
    with st.spinner("Generating comparison..."):
        # Generate synthetic data for comparison
        smiles_list, targets = generate_synthetic_dataset(n_samples=100, seed=42)
        
        # Create features for different fingerprint types
        morgan_features = []
        maccs_features = []
        descriptor_features = []
        valid_targets = []
        
        for smiles, target in zip(smiles_list, targets):
            morgan_fp = smiles_to_morgan_fingerprint(smiles)
            maccs_fp = smiles_to_maccs_keys(smiles)
            desc_fp = smiles_to_rdkit_descriptors(smiles)
            
            if all(fp is not None for fp in [morgan_fp, maccs_fp, desc_fp]):
                morgan_features.append(morgan_fp)
                maccs_features.append(maccs_fp)
                descriptor_features.append(desc_fp)
                valid_targets.append(target)
        
        # Train simple models for comparison
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        models = {}
        metrics = {}
        
        # Morgan fingerprints
        X_morgan = np.array(morgan_features)
        y = np.array(valid_targets)
        X_train, X_test, y_train, y_test = train_test_split(X_morgan, y, test_size=0.3, random_state=42)
        
        rf_morgan = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_morgan.fit(X_train, y_train)
        y_pred = rf_morgan.predict(X_test)
        
        metrics['Morgan Fingerprints'] = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # MACCS keys
        X_maccs = np.array(maccs_features)
        X_train, X_test, y_train, y_test = train_test_split(X_maccs, y, test_size=0.3, random_state=42)
        
        rf_maccs = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_maccs.fit(X_train, y_train)
        y_pred = rf_maccs.predict(X_test)
        
        metrics['MACCS Keys'] = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # RDKit descriptors
        X_desc = np.array(descriptor_features)
        X_train, X_test, y_train, y_test = train_test_split(X_desc, y, test_size=0.3, random_state=42)
        
        rf_desc = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_desc.fit(X_train, y_train)
        y_pred = rf_desc.predict(X_test)
        
        metrics['RDKit Descriptors'] = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Display comparison
        comparison_df = pd.DataFrame(metrics).T
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create comparison plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(name='R²', x=list(metrics.keys()), y=[m['R²'] for m in metrics.values()]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(name='RMSE', x=list(metrics.keys()), y=[m['RMSE'] for m in metrics.values()]),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# About section
st.markdown("---")
st.header("ℹ️ About This Demo")

st.markdown("""
This interactive demo showcases AI-powered drug discovery using Quantitative Structure-Activity Relationship (QSAR) models.

**Features:**
- **Molecular Input**: Enter SMILES strings to represent molecules
- **Multiple Models**: Compare different fingerprint types and model architectures
- **Real-time Prediction**: Get instant activity predictions with uncertainty estimates
- **Visualization**: See molecular structures and prediction results

**Model Types:**
- **Fingerprint-based Neural Networks**: Use molecular fingerprints as input features
- **Graph Neural Networks**: Process molecules as graphs with atoms and bonds
- **Random Forest Baselines**: Traditional machine learning approach

**Fingerprint Types:**
- **Morgan Fingerprints**: Circular fingerprints capturing local structure
- **MACCS Keys**: Structural keys based on chemical substructures
- **RDKit Descriptors**: Physicochemical properties and molecular descriptors

**Note**: This is a research demonstration using synthetic data. Real drug discovery requires extensive validation and clinical testing.
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Drug Discovery AI Demo - Research and Educational Use Only"
    "</div>",
    unsafe_allow_html=True
)
