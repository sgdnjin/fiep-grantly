# Grantly Dashboard
import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit as st
from collections import defaultdict
from fpdf import FPDF
import plotly.graph_objects as go

# --- Load data ---
df = pd.read_excel("EN_cleaned_with_loans.xlsx")
gdf_states = gpd.read_file("germany_states_en.geojson")

# --- Helper functions ---
def tokenize_series(s):
    return s.dropna().apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])

def count_tokens(series):
    counts = defaultdict(int)
    for items in tokenize_series(series):
        for token in items:
            counts[token] += 1
    return pd.DataFrame({'Category': list(counts.keys()), 'Count': list(counts.values())}).sort_values(by="Count", ascending=True)

def apply_filters(data, filters):
    def match_any(cell, selections):
        if not selections: return True
        if pd.isna(cell): return False
        tokens = [t.strip() for t in str(cell).split(',')]
        return any(sel in tokens for sel in selections)

    region_selection = filters["Region"]
    extended_region_selection = region_selection + ["Germany-wide"] if region_selection else []

    filtered = data[
        data["Funding type"].apply(lambda x: match_any(x, filters["Funding type"])) &
        data["Funding theme"].apply(lambda x: match_any(x, filters["Funding theme"])) &
        data["Eligible applicants"].apply(lambda x: match_any(x, filters["Eligible applicants"])) &
        data["Region"].apply(lambda x: match_any(x, extended_region_selection))
    ]
    return filtered

def generate_pdf(row):
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, row["Title"], ln=True, align="C")

        def body(self, row):
            self.set_font("Arial", "", 12)
            self.ln(10)
            self.multi_cell(0, 10, f"Description:\n{row['Text2']}\n", align="L")
            self.ln(5)
            self.cell(0, 10, f"Region: {row['Region']}", ln=True)
            self.cell(0, 10, f"Funding Type: {row['Funding type']}", ln=True)
            self.cell(0, 10, f"Funding Theme: {row['Funding theme']}", ln=True)
            self.cell(0, 10, f"Eligible Applicants: {row['Eligible applicants']}", ln=True)
            self.cell(0, 10, f"URL: {row['URL']}", ln=True, link=row["URL"])

    pdf = PDF()
    pdf.add_page()
    pdf.body(row)
    path = "program_summary.pdf"
    pdf.output(path)
    return path

# --- Streamlit config ---
st.set_page_config(page_title="Grantly Dashboard", layout="wide")
if st.session_state.get("reset_trigger"):
    st.session_state.clear()  # clear all filters and widgets
    st.session_state["loan_slider"] = 0  # re-initialize slider manually
    st.session_state["reset_trigger"] = False
    st.rerun()
st.title("Grantly — Government Funding Explorer")

# --- Sidebar filters ---
with st.sidebar:
    st.image("logo1.jpeg")
    st.markdown("## Filter Funding Programs")

    if st.button("\U0001F504 Reset filters"):
        # st.session_state.clear()
        st.session_state["reset_trigger"] = True
        st.rerun()

    filters = {
        "Funding type": st.multiselect(
            "Funding type",
            sorted(count_tokens(df["Funding type"])["Category"]),
            default=st.session_state.get("Funding type", []),
            key="Funding type"
        ),
        "Funding theme": st.multiselect(
            "Funding theme",
            sorted(count_tokens(df["Funding theme"])["Category"]),
            default=st.session_state.get("Funding theme", []),
            key="Funding theme"
        ),
        "Eligible applicants": st.multiselect(
            "Eligible applicants",
            sorted(count_tokens(df["Eligible applicants"])["Category"]),
            default=st.session_state.get("Eligible applicants", []),
            key="Eligible applicants"
        ),
        "Region": st.multiselect(
            "Region",
            sorted(set(token for sublist in tokenize_series(df["Region"]) for token in sublist if token != "Germany-wide")),
            default=st.session_state.get("Region", []),
            key="Region"
        )
    }

# --- Filtered data ---
filtered = apply_filters(df, filters)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Visual Summary", "Filtered Results", "Loan Explorer"])

# --- Tab 1 ---
with tab1:
    st.markdown(f"### Total Programs Matching Filters: {len(filtered)}")

    bar1, bar2 = st.columns([1, 1])
    with bar1:
        theme_df = count_tokens(filtered["Funding theme"])
        fig = px.bar(theme_df, x="Count", y="Category", orientation="h", title="Funding Theme", color="Category", color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="#0e1117", font=dict(color="white"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with bar2:
        entity_df = count_tokens(filtered["Eligible applicants"])
        fig = px.bar(entity_df, x="Count", y="Category", orientation="h", title="Eligible Applicants", color="Category", color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="#0e1117", font=dict(color="white"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    lower1, lower2 = st.columns([2, 1])
    with lower1:
        region_counts = tokenize_series(filtered["Region"]).explode().value_counts().rename_axis("Region").reset_index(name="Count")
        region_counts = region_counts[region_counts["Region"] != "Germany-wide"]
        gdf = gdf_states.merge(region_counts, left_on="VARNAME_1", right_on="Region", how="left").fillna({"Count": 0})
        map_fig = px.choropleth(
            gdf, geojson=gdf.geometry, locations=gdf.index, color="Count",
            hover_name="VARNAME_1", title="Funding Programs by Region",
            color_continuous_scale="Plasma"
        )
        map_fig.update_geos(fitbounds="locations", visible=False, projection_scale=6.5, center={"lat": 51.2, "lon": 10.5})
        map_fig.update_layout(
            template="plotly_dark", dragmode=False, width=500, height=500,
            margin=dict(l=0, r=0, t=30, b=0), coloraxis_colorbar=dict(x=0.87, len=0.75),
            geo=dict(
                lataxis_showgrid=False,
                lonaxis_showgrid=False,
                showland=True,
                landcolor="black"
            )
        )
        st.plotly_chart(map_fig, use_container_width=False)

    with lower2:
        ft_df = count_tokens(filtered["Funding type"])
        fig = px.pie(ft_df, names="Category", values="Count", hole=0.4, title="Funding Type", color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_traces(textinfo="none")
        fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0e1117", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2 ---
with tab2:
    st.markdown("## Filtered Results")
    if not filtered.empty:
        titles = filtered["Title"].dropna().unique()
        selected_title = st.selectbox("Choose a program", titles, label_visibility="collapsed", index=0)

        selected_row = filtered[filtered["Title"] == selected_title].iloc[0]

        st.subheader(selected_row["Title"])
        st.markdown(f"**Region:** {selected_row['Region']}")
        st.markdown(f"**Funding Type:** {selected_row['Funding type']}")
        st.markdown(f"**Funding Theme:** {selected_row['Funding theme']}")
        st.markdown(f"**Eligible Applicants:** {selected_row['Eligible applicants']}")
        st.markdown(f"**URL:** [Open Link]({selected_row['URL']})", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(selected_row["Text2"])

        pdf_path = generate_pdf(selected_row)
        with open(pdf_path, "rb") as f:
            st.download_button("\U0001F4C4 Download Summary as PDF", f, file_name="program_summary.pdf", mime="application/pdf")
    else:
        st.info("No programs match the current filters.")

# --- Tab 3: Loan Explorer ---
with tab3:
    st.markdown("## Loan Explorer")
    loan_df = filtered[filtered["Funding type"].str.contains("Loan", na=False)]

    valid_max_amounts = pd.to_numeric(loan_df["loan_max_amount"], errors="coerce").dropna()
    if not valid_max_amounts.empty:
        max_amount = valid_max_amounts.max()
        loan_amount = st.slider(
            "Required loan amount", min_value=0, max_value=int(max_amount), step=1000, 
            # value=0,
            format="€%s", label_visibility="visible", key="loan_slider"
        )
        loan_df = loan_df[pd.to_numeric(loan_df["loan_max_amount"], errors="coerce") >= loan_amount]
    else:
        loan_df = loan_df.copy()

    st.markdown(f"### Amount of loans currently selected: {len(loan_df)}")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        min_val = pd.to_numeric(loan_df["loan_min_amount"], errors="coerce").dropna().min()
        st.metric("Min Loan Amount", f"€{min_val:,.0f}" if pd.notna(min_val) else "N/A")
    with col2:
        max_val = pd.to_numeric(loan_df["loan_max_amount"], errors="coerce").dropna().max()
        st.metric("Max Loan Amount", f"€{max_val:,.0f}" if pd.notna(max_val) else "N/A")
    with col3:
        avg_pct = pd.to_numeric(loan_df["loan_percentage"], errors="coerce").dropna().mean()
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_pct if not pd.isna(avg_pct) else 0,
            title={'text': "Average % of eligible costs covered"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#fb9f3a"},
                'bgcolor': "#1e1e1e",
                'borderwidth': 1,
            }
        ))
        gauge.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(gauge, use_container_width=True)

    col4, col5 = st.columns([1, 1])
    with col4:
        violin_data_max = pd.to_numeric(loan_df["loan_max_amount"], errors="coerce").dropna()
        fig = px.violin(violin_data_max, box=True, points="all", title="Maximum Loan Amount", color_discrete_sequence=["#d8576b"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", font=dict(color="white"), xaxis_title="")
        fig.update_yaxes(title="Loan Amount (€)")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        violin_data = pd.to_numeric(loan_df["loan_percentage"], errors="coerce").dropna()
        if not violin_data.empty:
            fig = px.violin(violin_data, box=True, points="all", title="Eligible cost funding percentage", color_discrete_sequence=["#9c179e"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", font=dict(color="white"))
            fig.update_yaxes(title="% of project costs")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for loan percentage.")
