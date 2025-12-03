import pandas as pd
import streamlit as st

from wet_net.paths import RESULTS_DIR

results_path = RESULTS_DIR / "dummy_results.csv"

df = pd.read_csv(results_path)

st.title("Dummy Results Viewer")
st.dataframe(df)
