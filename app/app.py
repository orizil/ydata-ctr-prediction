import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pickle

import sys
sys.path.append('../')

from notebooks.entire_flow_test_data import process_test_data

st.title("CTR Prediction")
st.markdown("""
**Instructions:**
1. Upload a CSV file following the competition schema **(no target column)**.
2. Click "Generate Predictions".
3. Download the resulting CSV with **one column** containing probability scores (0 to 1).
""")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"],
    help="File should follow the competition schema, excluding the target column."
)

if uploaded_file is not None:
    # Read the uploaded CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Show a preview of the data
    st.write("Preview of uploaded data:", df.head())

    if st.button("Generate Predictions"):
        try:

            predictions = process_test_data(
                df=df,
            )

            # Create results DataFrame
            results_df = pd.DataFrame({'click_probability': predictions})

            # Show preview of predictions
            st.write("Preview of predictions:", results_df.head())

            # Download button
            st.download_button(
                label="Download Predictions",
                data=results_df.to_csv(index=False),
                file_name="ctr_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
else:
    st.info("Please upload a CSV file to generate predictions.")
