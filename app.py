import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Ito Approximation",
    layout="wide"
)

def b_motion_np(n, m, T, seeded=True):
    """Generate Brownian motion paths"""
    dt = T/n  # partition the total time interval into equally spaced intervals of length dt
    sig = 1
    vol = sig*np.sqrt(dt)
    if seeded:
        seed = 1234
        rg = np.random.RandomState(seed) 
        incs = rg.standard_normal(size=(n, m))
    else:
        incs = np.random.standard_normal(size=(n, m))
                                          
    Bt = np.concatenate((np.zeros((1, m)), incs), axis=0).cumsum(axis=0) 
    Bt *= vol
    tline = np.linspace(0, T, n+1)                            
    t = np.repeat(tline.reshape((n+1, 1)), m, axis=1)   
    return Bt, t

def create_ito_plots(n, T):
    """Create Ito approximation plots"""
    m = 1  # simulate max pre-set paths, user is just slicing them
    Bt, t = b_motion_np(n=n, m=m, T=T, seeded=True) 
    inc = np.diff(Bt, axis=0)
    Q = (Bt[1:n + 1] - Bt[0:n])**2

    G = lambda x: np.exp(x)
    DG = lambda x: np.exp(x)
    DDG = lambda x: np.exp(x)
    
    # Build G'(x)dB
    GdB = np.zeros(shape=(n, m))
    GdB[0] = np.repeat(DG(0), m) * inc[0]
    GdB[1:] = DG(Bt[0:n - 1]) * inc[1:]
    SI = np.zeros(shape=(n, m))
    # Stochastic Integral is cumulative sum of G'(B)dB plus initial
    QVV = np.zeros(shape=(n, m))
    QVV = 0.5 * (DDG(Bt[0:n])*Q).cumsum(axis=0)
    SI = GdB.cumsum(axis=0) + QVV + G(0)
    
    # First-order approximation (without second derivative term)
    SI_first_order = GdB.cumsum(axis=0) + G(0)

    # Main comparison plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=t[1:, 0],
        y=G(Bt[1:n + 1, 0]),
        mode="lines",
        name=r'Exact Solution: $\mathrm{e}^B_t$',
        line=dict(color='blue', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=t[1:, 0], 
        y=SI_first_order[:, 0], 
        mode="lines", 
        name="First-Order Approximation",
        line=dict(color='green', width=2, dash='dot')
    ))
    
    fig1.add_trace(go.Scatter(
        x=t[1:, 0], 
        y=SI[:, 0], 
        mode="lines", 
        name="Ito Approximation",
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig1.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        height=400
    )

    # Second derivative term plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=t[1:, 0],
        y=QVV[:, 0],
        mode="lines",
        line=dict(color='green', width=2)
    ))

    fig2.update_layout(
        xaxis_title="Time",
        yaxis_title="Second Derivative Term",
        template="plotly_white",
        showlegend=False,
        height=400
    )

    return fig1, fig2

# Top control area
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.selectbox(
            "Number of time subdivisions:",
            options=[10, 100, 1000, 10000],
            index=1
        )
    
    with col2:
        time_horizon = st.slider(
            "Time horizon:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    

# Generate and display plots
fig1, fig2 = create_ito_plots(n, time_horizon)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)


