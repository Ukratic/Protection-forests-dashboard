import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pydeck as pdk
import openpyxl

### Config
st.set_page_config(
    page_title="Protection Forest Project",
    layout="wide"
)

### App
st.title('Descriptive machine learning')
st.sidebar.success("Select a page above")

st.markdown("")

@st.cache(allow_output_mutation=True)
def load_data(nrows):
    score_model = pd.read_excel('https://storage.googleapis.com/forest_dashboard/score_total.xlsx')
    return score_model

data_load_state = st.text('Loading data, please wait...')
score_model = load_data(None)
data_load_state.text("")


score_model["Score"] = round(score_model["Score"], 2)
desc = score_model.loc[score_model["Mode"] == "Descriptive",:]
desc['simple_name'] = desc['Model'].apply(lambda x: "LR" if x == "Linear Regression"
else "Lasso" if x == "Lasso"
else "RF" if x == "Random Forest"
else "HGBr" if x == "Hist Gradient Boost Regressor"
else "DT" if x == "Decision Tree"
else "Voting" if x == "Voting of differents models"
else "RC" if x == "Ridge Classifier"
else "CNB" if x == "Categorical Naive Bayes"
else "HGBc" if x == "Hist Gradient Boost Classifier"
else x)

st.header("Descriptive machine learning scores for basal area")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(x="simple_name", y = "Score", data_frame=desc.loc[desc["Target"] == "Basal Area (m2 / ha)",:],
             labels={"simple_name":"model tested", "Score":"determination coefficient"},
             hover_data=['Score'], color='Score', color_continuous_scale='Spectral',
             text_auto=True, width=1000)
    fig1.update_coloraxes(showscale=False)
    fig1.update(layout_showlegend=False)
    fig1.update_yaxes(range=[0,1])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown('LR = Linear Regression (R²)')
    st.markdown('Lasso = Lasso (R²)')
    st.markdown('RF = Random Forest (R²)')
    st.markdown('HGBr = Hist Gradient Boost Regressor (R²)')
    st.markdown('DT = Decision Tree (R²)')
    st.markdown('Voting = Voting of differents models (R²)')



st.header("Descriptive machine learning scores for growth")
col3, col4 = st.columns(2)
with col3:
    fig2 = px.bar(x="simple_name", y = "Score", data_frame=desc.loc[desc["Target"] == "Growth",:],
             labels={"simple_name":"model tested", "Score":"determination coefficient"},
             category_orders={"simple_name": ["LR", "Lasso", "DT", "RF", "HGBr"]},
             hover_data=['Score'], color='Score', color_continuous_scale='Spectral',
             text_auto=True, width=1000)
    fig2.update_coloraxes(showscale=False)
    fig2.update(layout_showlegend=False)
    fig2.update_yaxes(range=[0,1])
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    st.markdown('LR = Linear Regression (R²)')
    st.markdown('Lasso = Lasso (R²)')
    st.markdown('DT = Decision Tree (R²)')
    st.markdown('RF = Random Forest (R²)')
    st.markdown('HGBr = Hist Gradient Boost Regressor (R²)')



st.header("Descriptive machine learning scores for regeneration coverage rate")
col5, col6 = st.columns(2)
with col5:
    fig3 = px.bar(x="simple_name", y = "Score", data_frame=desc.loc[desc["Target"] == "Regeneration coverate rate",:],
             labels={"simple_name":"model tested", "Score":"Performance"},
             hover_data=['Score'], color='Score', color_continuous_scale='Spectral',
             text_auto=True, width=1000)
    fig3.update_coloraxes(showscale=False)
    fig3.update(layout_showlegend=False)
    fig3.update_yaxes(range=[0,1])
    st.plotly_chart(fig3, use_container_width=True)

with col6:
    st.markdown('Lasso = Lasso (R²)')
    st.markdown('RC = RidgeClassifier (Accuracy)')
    st.markdown('CNB = Categorical Naive Bayes (F1)')
    st.markdown('HGBc = Hist Gradient Boost Classifier (F1)')
    st.markdown('RF = Random Forest (R²)')
    st.markdown('HGBr = Hist Gradient Boost Regressor (R²)')
