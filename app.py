import altair as alt
import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


# -------------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Portfolio Dashboard",
    layout="wide",
)

st.title("Credit Risk Portfolio Dashboard")
st.caption(
    "XGBoost credit-risk scoring, portfolio expected loss, stress testing, "
    "and borrower-level risk analysis."
)


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------
DATA_PATH = "lending_club_100k.csv"
PREPROCESSOR_PATH = "credit_risk_preprocessor.pkl"
MODEL_PATH = "xgb_credit_risk_model.json"

LGD = 0.60

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
]

RISK_ORDER = [
    "Low Risk",
    "Moderate Risk",
    "High Risk",
    "Very High Risk",
]


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the fitted preprocessor and trained XGBoost model once."""
    fitted_preprocessor = joblib.load(PREPROCESSOR_PATH)

    fitted_model = XGBClassifier()
    fitted_model.load_model(MODEL_PATH)

    return fitted_preprocessor, fitted_model


@st.cache_data
def load_data():
    """Load the Lending Club sample once and reuse it across app reruns."""
    return pd.read_csv(DATA_PATH)


def assign_risk_level(pd_value):
    """Convert a predicted probability of default into a dashboard risk band."""
    if pd_value < 0.10:
        return "Low Risk"
    if pd_value < 0.20:
        return "Moderate Risk"
    if pd_value < 0.30:
        return "High Risk"
    return "Very High Risk"


# -------------------------------------------------------------------
# LOAD MODEL AND DATA
# -------------------------------------------------------------------
try:
    preprocessor, model = load_model()
    df = load_data()
except Exception as error:
    st.error("The model or dataset failed to load.")
    st.exception(error)
    st.stop()


# Keep the same completed-loan population used for model development.
portfolio_df = df[
    df["loan_status"].isin(["Fully Paid", "Charged Off"])
].copy()

# Score the completed-loan portfolio.
X_portfolio = portfolio_df[FEATURES]
X_portfolio_transformed = preprocessor.transform(X_portfolio)

portfolio_df["PD"] = model.predict_proba(
    X_portfolio_transformed
)[:, 1]

# Credit-risk assumptions used throughout the project.
portfolio_df["LGD"] = LGD
portfolio_df["EAD"] = portfolio_df["loan_amnt"]

portfolio_df["Expected_Loss"] = (
    portfolio_df["PD"]
    * portfolio_df["LGD"]
    * portfolio_df["EAD"]
)


# -------------------------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------------------------
overview_tab, stress_tab, borrower_tab = st.tabs(
    [
        "Portfolio Overview",
        "Stress Testing",
        "Borrower Analysis",
    ]
)

with overview_tab:
    # -------------------------------------------------------------------
    # PORTFOLIO OVERVIEW
    # -------------------------------------------------------------------
    st.header("Portfolio Overview")

    st.caption(
        f"Portfolio contains {len(portfolio_df):,} completed loans "
        f"from a {len(df):,}-loan sample."
    )

    total_exposure = portfolio_df["EAD"].sum()
    average_pd = portfolio_df["PD"].mean()
    total_expected_loss = portfolio_df["Expected_Loss"].sum()
    expected_loss_rate = total_expected_loss / total_exposure

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

    overview_col1.metric(
        "Total Exposure",
        f"${total_exposure / 1_000_000:,.1f}M",
    )

    overview_col2.metric(
        "Average Probability of Default (PD)",
        f"{average_pd:.2%}",
    )

    overview_col3.metric(
        "Expected Loss",
        f"${total_expected_loss / 1_000_000:,.1f}M",
    )

    overview_col4.metric(
        "Expected Loss Rate",
        f"{expected_loss_rate:.2%}",
    )

    st.divider()


    # -------------------------------------------------------------------
    # PORTFOLIO RISK SEGMENTATION
    # -------------------------------------------------------------------
    st.header("Portfolio Risk Segmentation")

    portfolio_df["Risk Level"] = pd.cut(
        portfolio_df["PD"],
        bins=[0, 0.10, 0.20, 0.30, 1.00],
        labels=RISK_ORDER,
        include_lowest=True,
    )

    risk_summary = (
        portfolio_df
        .groupby("Risk Level", observed=True)
        .agg(
            Loans=("PD", "count"),
            Exposure=("EAD", "sum"),
            Average_PD=("PD", "mean"),
            Expected_Loss=("Expected_Loss", "sum"),
        )
        .reset_index()
    )

    # Make a display-only copy so the underlying numeric values stay usable
    # for charts and calculations.
    risk_display = risk_summary.copy()

    risk_display["Loans"] = risk_display["Loans"].map(
        lambda value: f"{value:,}"
    )

    risk_display["Exposure"] = risk_display["Exposure"].map(
        lambda value: f"${value / 1_000_000:,.1f}M"
    )

    risk_display["Average_PD"] = risk_display["Average_PD"].map(
        lambda value: f"{value:.2%}"
    )

    risk_display["Expected_Loss"] = risk_display["Expected_Loss"].map(
        lambda value: f"${value / 1_000_000:,.1f}M"
    )

    st.dataframe(
        risk_display,
        use_container_width=True,
        hide_index=True,
    )

    # Display the two portfolio charts side by side on wider screens.
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Exposure by Risk Level")

        exposure_chart = alt.Chart(risk_summary).mark_bar().encode(
            x=alt.X(
                "Risk Level:N",
                sort=RISK_ORDER,
                title="Risk Level",
            ),
            y=alt.Y(
                "Exposure:Q",
                title="Exposure ($M)",
                axis=alt.Axis(format="~s"),
            ),
            tooltip=[
                alt.Tooltip("Risk Level:N", title="Risk Level"),
                alt.Tooltip(
                    "Exposure:Q",
                    title="Exposure",
                    format="$,.0f",
                ),
            ],
        )

        st.altair_chart(
            exposure_chart,
            use_container_width=True,
        )

    with chart_col2:
        st.subheader("Expected Loss by Risk Level")

        expected_loss_chart = alt.Chart(risk_summary).mark_bar().encode(
            x=alt.X(
                "Risk Level:N",
                sort=RISK_ORDER,
                title="Risk Level",
            ),
            y=alt.Y(
                "Expected_Loss:Q",
                title="Expected Loss ($M)",
                axis=alt.Axis(format="~s"),
            ),
            tooltip=[
                alt.Tooltip("Risk Level:N", title="Risk Level"),
                alt.Tooltip(
                    "Expected_Loss:Q",
                    title="Expected Loss",
                    format="$,.0f",
                ),
            ],
        )

        st.altair_chart(
            expected_loss_chart,
            use_container_width=True,
        )

    st.divider()

with stress_tab:
    # -------------------------------------------------------------------
    # INTERACTIVE STRESS TEST
    # -------------------------------------------------------------------
    st.header("Interactive Stress Test")

    st.caption(
        "Increase modeled probability of default while holding Loss Given Default (LGD) and Exposure at Default (EAD) "
        "constant. Stressed Probability of Default (PD) is capped at 100%."
    )

    pd_multiplier = st.slider(
        "Probability of Default (PD) Stress Multiplier",
        min_value=1.00,
        max_value=2.00,
        value=1.25,
        step=0.05,
    )

    portfolio_df["Stressed_PD"] = (
        portfolio_df["PD"] * pd_multiplier
    ).clip(upper=1)

    portfolio_df["Stressed_Expected_Loss"] = (
        portfolio_df["Stressed_PD"]
        * portfolio_df["LGD"]
        * portfolio_df["EAD"]
    )

    stressed_expected_loss = portfolio_df["Stressed_Expected_Loss"].sum()
    increase_in_loss = stressed_expected_loss - total_expected_loss
    percent_increase = increase_in_loss / total_expected_loss

    stress_col1, stress_col2, stress_col3 = st.columns(3)

    stress_col1.metric(
        "Stressed Expected Loss",
        f"${stressed_expected_loss / 1_000_000:,.1f}M",
    )

    stress_col2.metric(
        "Increase in Loss",
        f"${increase_in_loss / 1_000_000:,.1f}M",
    )

    stress_col3.metric(
        "Percent Increase",
        f"{percent_increase:.2%}",
    )

    stress_chart_df = pd.DataFrame(
        {
            "Scenario": ["Baseline", "Stressed"],
            "Expected Loss": [
                total_expected_loss,
                stressed_expected_loss,
            ],
        }
    )

    stress_chart = alt.Chart(stress_chart_df).mark_bar().encode(
        x=alt.X(
            "Scenario:N",
            title="Scenario",
        ),
        y=alt.Y(
            "Expected Loss:Q",
            title="Expected Loss ($M)",
            axis=alt.Axis(format="~s"),
        ),
        tooltip=[
            alt.Tooltip("Scenario:N", title="Scenario"),
            alt.Tooltip(
                "Expected Loss:Q",
                title="Expected Loss",
                format="$,.0f",
            ),
        ],
    )

    st.altair_chart(
        stress_chart,
        use_container_width=True,
    )

    st.divider()

with borrower_tab:
    # -------------------------------------------------------------------
    # INDIVIDUAL BORROWER RISK PREDICTION
    # -------------------------------------------------------------------
    st.header("Individual Borrower Risk Prediction")

    st.caption(
        "Enter borrower and loan characteristics to estimate probability of "
        "default and expected loss."
    )

    # A form keeps Streamlit from rerunning the prediction section every time
    # one borrower input is changed.
    with st.form("borrower_risk_form"):
        input_col1, input_col2, input_col3 = st.columns(3)

        with input_col1:
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=50000,
                value=15000,
                step=500,
            )

            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=40.0,
                value=12.0,
                step=0.1,
            )

            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=65000.0,
                step=1000.0,
            )

            grade = st.selectbox(
                "Grade",
                sorted(portfolio_df["grade"].dropna().unique()),
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                sorted(portfolio_df["home_ownership"].dropna().unique()),
            )

        with input_col2:
            term = st.selectbox(
                "Term",
                ["36 months", "60 months"],
            )

            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=0.0,
                value=400.0,
                step=25.0,
            )

            dti = st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=100.0,
                value=18.0,
                step=0.5,
            )

            # Restrict sub-grade choices to the grade selected above.
            sub_grade_options = sorted(
                portfolio_df.loc[
                    portfolio_df["grade"] == grade,
                    "sub_grade",
                ]
                .dropna()
                .unique()
            )

            sub_grade = st.selectbox(
                "Sub Grade",
                sub_grade_options,
            )

            verification_status = st.selectbox(
                "Verification Status",
                sorted(
                    portfolio_df["verification_status"]
                    .dropna()
                    .unique()
                ),
            )

        with input_col3:
            fico_range_low = st.number_input(
                "FICO Range Low",
                min_value=300,
                max_value=850,
                value=680,
                step=5,
            )

            fico_range_high = st.number_input(
                "FICO Range High",
                min_value=300,
                max_value=850,
                value=684,
                step=5,
            )

            revol_util = st.number_input(
                "Revolving Utilization (%)",
                min_value=0.0,
                max_value=150.0,
                value=40.0,
                step=1.0,
            )

            emp_length = st.selectbox(
                "Employment Length",
                sorted(
                    portfolio_df["emp_length"]
                    .dropna()
                    .astype(str)
                    .unique()
                ),
            )

            purpose = st.selectbox(
                "Loan Purpose",
                sorted(portfolio_df["purpose"].dropna().unique()),
            )

        predict_button = st.form_submit_button(
            "Predict Default Risk",
            use_container_width=True,
        )


    # Only show borrower results after the user submits the form.
    if predict_button:
        borrower_data = pd.DataFrame(
            {
                "loan_amnt": [loan_amnt],
                "term": [term],
                "int_rate": [int_rate],
                "installment": [installment],
                "grade": [grade],
                "sub_grade": [sub_grade],
                "emp_length": [emp_length],
                "home_ownership": [home_ownership],
                "annual_inc": [annual_inc],
                "verification_status": [verification_status],
                "purpose": [purpose],
                "dti": [dti],
                "fico_range_low": [fico_range_low],
                "fico_range_high": [fico_range_high],
                "revol_util": [revol_util],
            }
        )

        borrower_transformed = preprocessor.transform(
            borrower_data
        )

        borrower_pd = model.predict_proba(
            borrower_transformed
        )[:, 1][0]

        risk_level = assign_risk_level(borrower_pd)

        borrower_expected_loss = (
            borrower_pd
            * LGD
            * loan_amnt
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Predicted PD",
            f"{borrower_pd:.2%}",
        )

        result_col2.metric(
            "Risk Level",
            risk_level,
        )

        result_col3.metric(
            "Expected Loss",
            f"${borrower_expected_loss:,.2f}",
        )

        # ---------------------------------------------------------------
        # BORROWER VS. PORTFOLIO COMPARISON
        # ---------------------------------------------------------------
        st.subheader("Borrower Risk Drivers")
        st.caption(
            "Descriptive comparison with portfolio averages. "
            "The notebook contains the full SHAP model explanation."
        )

        comparison_df = pd.DataFrame(
            {
                "Feature": [
                    "Interest Rate",
                    "DTI",
                    "FICO Score",
                    "Annual Income",
                    "Revolving Utilization",
                ],
                "Borrower": [
                    int_rate,
                    dti,
                    fico_range_low,
                    annual_inc,
                    revol_util,
                ],
                "Portfolio Average": [
                    portfolio_df["int_rate"].mean(),
                    portfolio_df["dti"].mean(),
                    portfolio_df["fico_range_low"].mean(),
                    portfolio_df["annual_inc"].mean(),
                    portfolio_df["revol_util"].mean(),
                ],
            }
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Borrower": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Portfolio Average": st.column_config.NumberColumn(
                    format="%.2f"
                ),
            },
        )

        risk_flags = []

        if int_rate > portfolio_df["int_rate"].mean():
            risk_flags.append(
                "Interest rate is above the portfolio average."
            )

        if dti > portfolio_df["dti"].mean():
            risk_flags.append(
                "DTI is above the portfolio average."
            )

        if fico_range_low < portfolio_df["fico_range_low"].mean():
            risk_flags.append(
                "FICO score is below the portfolio average."
            )

        if revol_util > portfolio_df["revol_util"].mean():
            risk_flags.append(
                "Revolving utilization is above the portfolio average."
            )

        if annual_inc < portfolio_df["annual_inc"].mean():
            risk_flags.append(
                "Annual income is below the portfolio average."
            )

        if risk_flags:
            st.write("**Key risk indicators:**")

            for flag in risk_flags:
                st.write(f"- {flag}")
        else:
            st.success(
                "No major risk indicators are elevated relative "
                "to the portfolio average."
            )
