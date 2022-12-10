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

st.markdown("""The Histogram Gradient Boost Regressor, our best model, is a Decision Tree. To begin with, I should point out that Decision Trees and Random Forest generally are a good place to start with the type of data we had, a mix of numerical and categorical features.
Now, Decision Trees are often outperformed by Random Forest and a few other algorithms, but there are several reasons why it performed so well.

First, this is not just a simple decision tree with optimized hyperparameters through gridsearch (though we did do that). As its name implies it uses boosting (gradient boosting specifically), which basically means that the model is improved sequentially, not through bagging (and therefore parallel training) as it would with Random Forest.
Second, these models (histogram-based) have very fast execution and it is therefore really easy to "play around" with hyperparameters and different features and optimize every time for meaningful evaluation.
Third, native support for missing data means that it makes the preprocessing effort much easier : a significant point when planning to try out different processes in a very short time-frame with 100+ variables, some quite hard to explain, as was our case.

The first caveat is that it can be hard to deal with overfitting when using boosting. This was indeed something of a conundrum on our attempts at modeling with less data.
And lastly, extracting feature importance required relying on either Game Theory's Shapley values, or a separate (not Histogram-based) Gradient Boosting Regressor, which is of course not a completely reliable approach.

Given more time, we would have switched to XGBoost's implementation to see if it would improve our results slightly and help us better identify important features.
Despite poor initial results, there might still also be an interesting way forward with the bayesian approach ( only for our classification task, of course), which we didn't have time to fully explore.""")


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
