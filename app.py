import os
import io
import tempfile
import streamlit as st
import pandas as pd

from households_fitting_notebook import run_pipeline

st.set_page_config(page_title="Households fitting (Option B)", layout="wide")

st.title("Households fitting — Option B")
st.write("Upload your fifteen-minute household workbook and run the full fitting pipeline.")

uploaded = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx"])
units_label = st.selectbox("Units of the fifteen-minute values", ["kilowatt (average over quarter hour)", "kilowatt hour (energy per quarter hour)"])
units = "kw" if units_label.startswith("kilowatt (average") else "kwh"
strict = st.checkbox("Require exactly four quarters per hour (drop incomplete hours)")
make_plots = st.checkbox("Generate diagnostic plots (histograms and quantile–quantile)")

run = st.button("Run fitting")

if run and uploaded is None:
    st.error("Please upload a workbook first.")

if run and uploaded is not None:
    # Save upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Capture printed diagnostics to show inside Streamlit
    import contextlib, sys
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        profiles, day_scalers, residuals = run_pipeline(
            tmp_path, sheet=None, units=units, outdir="out_households", require_full_hours=strict, make_plots=make_plots
        )
    st.success("Fitting completed.")

    with st.expander("Diagnostics log (text)"):
        st.text(buf.getvalue())

    st.subheader("Download outputs")
    st.download_button("Download profiles_households.csv",
                       data=profiles.to_csv(index=False), file_name="profiles_households.csv", mime="text/csv")
    st.download_button("Download day_scalers_households.csv",
                       data=day_scalers.to_csv(index=False), file_name="day_scalers_households.csv", mime="text/csv")
    st.download_button("Download residuals_households.csv",
                       data=residuals.to_csv(index=False), file_name="residuals_households.csv", mime="text/csv")

    st.subheader("Preview")
    c1, c2, c3 = st.columns(3)
    with c1: st.write("profiles_households"); st.dataframe(profiles.head(12))
    with c2: st.write("day_scalers_households"); st.dataframe(day_scalers)
    with c3: st.write("residuals_households"); st.dataframe(residuals)

    if make_plots and os.path.isdir("diagnostics"):
        st.subheader("Diagnostic plots")
        imgs = sorted([f for f in os.listdir("diagnostics") if f.endswith(".png")])
        for f in imgs:
            st.image(os.path.join("diagnostics", f), caption=f, use_container_width=True)
