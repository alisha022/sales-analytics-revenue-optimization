import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AAL Sales Analytics — Q4 2020",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('AusApparalSales4thQrt2020.csv')
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month
    return df

df = load_data()

# ─────────────────────────────────────────────
# STATE SUMMARY
# ─────────────────────────────────────────────
state_sales = (
    df.groupby('State')['Sales']
    .agg(Total_Sales='sum', Avg_Sales='mean', Transactions='count')
    .sort_values('Total_Sales', ascending=False)
    .reset_index()
)
state_sales['Revenue_Share_%'] = (
    state_sales['Total_Sales'] / state_sales['Total_Sales'].sum() * 100
).round(2)
state_sales['Total_Sales_M'] = (state_sales['Total_Sales'] / 1e6).round(2)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📊 AAL Regional Sales Analytics — Q4 2020")
st.markdown(
    "**Australian Apparel Limited** · Exploratory analysis of 7,560 sales transactions "
    "across Australian states · October – December 2020"
)
st.divider()

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────
total_rev   = df['Sales'].sum()
total_txn   = len(df)
top_state   = state_sales.iloc[0]['State']
top_share   = state_sales.iloc[0]['Revenue_Share_%']
top3_share  = state_sales.head(3)['Revenue_Share_%'].sum().round(2)
best_month  = df.groupby('Month')['Sales'].sum().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Q4 Revenue",    f"${total_rev/1e6:.1f}M")
col2.metric("Total Transactions",  f"{total_txn:,}")
col3.metric("Top State",           f"{top_state} ({top_share}%)")
col4.metric("Peak Month",          best_month)

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR FILTER
# ─────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
all_states = sorted(df['State'].unique())
selected_states = st.sidebar.multiselect(
    "Select States", all_states, default=all_states
)
selected_groups = st.sidebar.multiselect(
    "Select Customer Groups", sorted(df['Group'].unique()), default=sorted(df['Group'].unique())
)
selected_times = st.sidebar.multiselect(
    "Select Time of Day", ['Morning', 'Afternoon', 'Evening'], default=['Morning', 'Afternoon', 'Evening']
)

filtered_df = df[
    df['State'].isin(selected_states) &
    df['Group'].isin(selected_groups) &
    df['Time'].isin(selected_times)
]

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# SECTION 1 — STATE REVENUE
# ─────────────────────────────────────────────
st.subheader("1 · Revenue by State")

fs = (
    filtered_df.groupby('State')['Sales']
    .agg(Total_Sales='sum', Avg_Sales='mean', Transactions='count')
    .sort_values('Total_Sales', ascending=False)
    .reset_index()
)
fs['Revenue_Share_%'] = (fs['Total_Sales'] / fs['Total_Sales'].sum() * 100).round(2)
fs['Total_Sales_M'] = (fs['Total_Sales'] / 1e6).round(2)

col_a, col_b = st.columns([3, 2])

with col_a:
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [
        '#2ecc71' if s == fs['Total_Sales'].max() else
        '#e74c3c' if s == fs['Total_Sales'].min() else '#3498db'
        for s in fs['Total_Sales']
    ]
    bars = ax.bar(fs['State'], fs['Total_Sales_M'], color=colors, edgecolor='white', linewidth=0.7)
    for bar, share in zip(bars, fs['Revenue_Share_%']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{share}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Total Sales by State (AUD Million)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('State'); ax.set_ylabel('Sales (AUD Million)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("🟢 Highest Revenue  ·  🔴 Lowest Revenue  ·  🔵 Others")

with col_b:
    st.dataframe(
        fs[['State', 'Total_Sales_M', 'Revenue_Share_%', 'Transactions']]
        .rename(columns={
            'Total_Sales_M': 'Revenue (AUD M)',
            'Revenue_Share_%': 'Share %',
            'Transactions': 'Txn Count'
        }),
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ─────────────────────────────────────────────
# SECTION 2 — CUSTOMER GROUP ANALYSIS
# ─────────────────────────────────────────────
st.subheader("2 · Sales by Customer Group per State")

col_c, col_d = st.columns(2)

with col_c:
    group_state = filtered_df.groupby(['State', 'Group'])['Sales'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    group_state.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white', linewidth=0.5)
    ax.set_title('Sales by Customer Group per State', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('State'); ax.set_ylabel('Sales (AUD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    ax.legend(title='Customer Group', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_d:
    pivot = filtered_df.pivot_table(values='Sales', index='State', columns='Group', aggfunc='sum')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot / 1e6, annot=True, fmt='.1f', cmap='YlGnBu',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Sales (AUD M)'})
    ax.set_title('Heatmap: State vs Customer Group (AUD M)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ─────────────────────────────────────────────
# SECTION 3 — TIME OF DAY
# ─────────────────────────────────────────────
st.subheader("3 · Sales by Time of Day per State")

time_order_all = [t for t in ['Morning', 'Afternoon', 'Evening'] if t in selected_times]
if time_order_all:
    time_state = filtered_df.groupby(['State', 'Time'])['Sales'].sum().unstack()
    time_state = time_state[[t for t in time_order_all if t in time_state.columns]]

    fig, ax = plt.subplots(figsize=(10, 5))
    time_state.plot(kind='bar', ax=ax,
                    color=['#f39c12', '#3498db', '#9b59b6'][:len(time_order_all)],
                    edgecolor='white', linewidth=0.5)
    ax.set_title('Sales by Time of Day per State', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('State'); ax.set_ylabel('Sales (AUD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    ax.legend(title='Time Slot', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("💡 **Insight:** Evening shopping shows consistently strong performance — "
            "evening flash sales or extended store hours could drive incremental revenue.")

st.divider()

# ─────────────────────────────────────────────
# SECTION 4 — MONTHLY TRENDS
# ─────────────────────────────────────────────
st.subheader("4 · Monthly Sales Trend")

col_e, col_f = st.columns(2)

with col_e:
    month_order = ['October', 'November', 'December']
    available_months = [m for m in month_order if m in filtered_df['Month'].unique()]
    if available_months:
        monthly_state = filtered_df.groupby(['State', 'Month'])['Sales'].sum().unstack()
        monthly_state = monthly_state[[m for m in available_months if m in monthly_state.columns]]
        fig, ax = plt.subplots(figsize=(8, 5))
        monthly_state.plot(kind='line', ax=ax, marker='o', linewidth=2.5, markersize=8)
        ax.set_title('Monthly Sales by State', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('State'); ax.set_ylabel('Sales (AUD)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
        ax.legend(title='Month', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with col_f:
    monthly_total = (
        filtered_df.groupby(['Month_Num', 'Month'])['Sales']
        .sum().reset_index().sort_values('Month_Num')
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(monthly_total['Month'], monthly_total['Sales'] / 1e6,
           color=['#e67e22', '#27ae60', '#c0392b'], edgecolor='white')
    ax.set_title('Total Sales per Month', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Sales (AUD Million)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
    for bar, val in zip(ax.patches, monthly_total['Sales']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'${val/1e6:.1f}M', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.info("💡 **Insight:** December records the highest sales across all states — "
        "a strong signal to amplify festive season campaigns and pre-order promotions.")

st.divider()

# ─────────────────────────────────────────────
# SECTION 5 — DISTRIBUTION
# ─────────────────────────────────────────────
st.subheader("5 · Revenue Distribution")

col_g, col_h = st.columns(2)

with col_g:
    state_order = fs['State'].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=filtered_df, x='State', y='Sales', order=state_order,
                palette='coolwarm', ax=ax, linewidth=1.2)
    ax.set_title('Sales Distribution per State', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('State (sorted by total revenue)')
    ax.set_ylabel('Sales per Transaction (AUD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_h:
    fig, ax = plt.subplots(figsize=(7, 7))
    wedge_props = dict(width=0.55, edgecolor='white', linewidth=2)
    ax.pie(fs['Total_Sales'],
           labels=fs['State'],
           autopct='%1.1f%%',
           startangle=140,
           wedgeprops=wedge_props,
           colors=sns.color_palette('Set3', len(fs)))
    ax.set_title('Revenue Share by State', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ─────────────────────────────────────────────
# SECTION 6 — KEY RECOMMENDATIONS
# ─────────────────────────────────────────────
st.subheader("6 · Key Business Recommendations")

top3    = fs.head(3)['State'].tolist()
bottom3 = fs.tail(3)['State'].tolist()

rec_data = {
    "Action":  [
        "Invest & Expand",
        "Targeted Campaigns",
        "Evening Flash Sales",
        "Festive Push",
        "Loyalty Programs",
        "Segment Focus"
    ],
    "Target": [
        f"Top states: {', '.join(top3)}",
        f"Lower states: {', '.join(bottom3)}",
        "Low-revenue states",
        "All states",
        "All states",
        "Lower-revenue states"
    ],
    "Detail": [
        "Open additional outlets; increase SKU diversity",
        "Geo-targeted digital ads; influencer tie-ups",
        "Drive footfall during peak evening hours",
        "Amplify December promotions; pre-order campaigns",
        "Reward repeat customers across all age groups",
        "Prioritize best-performing customer group per state"
    ]
}

st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

st.divider()
st.caption("Analysis · AAL Sales & Marketing Department · Q4 2020 (Oct–Dec 2020) · Built with Streamlit")
