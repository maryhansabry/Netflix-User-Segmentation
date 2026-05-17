"""
config.py — All constants, cluster metadata, countries/regions
"""

# ── Cluster metadata ───────────────────────────────────────────────
CLUSTER_NAMES  = {0: "Active Heavy Watchers", 1: "Inactive Users", 2: "Casual Users"}
CLUSTER_ICONS  = {"Active Heavy Watchers": "🔥", "Inactive Users": "😴", "Casual Users": "📺"}
CLUSTER_BADGE  = {"Active Heavy Watchers": "badge-active", "Inactive Users": "badge-inactive", "Casual Users": "badge-casual"}
CLUSTER_COLOR  = {"Active Heavy Watchers": "#4ade80", "Inactive Users": "#f87171", "Casual Users": "#60a5fa"}
CLUSTER_PALETTE = {"Active Heavy Watchers": "#4ade80", "Inactive Users": "#f87171", "Casual Users": "#60a5fa"}

CLUSTER_DESC = {
    "Active Heavy Watchers": "High watch time and frequent logins. Your most loyal and engaged audience — perfect for premium content recommendations and loyalty rewards.",
    "Inactive Users": "Long time since last login and low engagement. High churn risk — needs re-engagement campaigns and win-back offers.",
    "Casual Users": "Moderate watch time with irregular activity. Personalized recommendations and trending content can boost their engagement.",
}

CLUSTER_RECOS = {
    "Active Heavy Watchers": [
        "🎬 Offer early access to new releases",
        "⭐ Upsell to higher-tier or annual plan",
        "🎁 Invite to loyalty & referral program",
    ],
    "Inactive Users": [
        "💌 Send a 'We miss you' email with personalized picks",
        "💰 Trigger a discount or 1-month free trial",
        "🆕 Highlight new popular titles since last visit",
    ],
    "Casual Users": [
        "🎯 Push personalized genre recommendations",
        "📈 Promote bingeable series & trending content",
        "🔔 Suggest a watch-list & reminder feature",
    ],
}

# ── Regions & Countries ────────────────────────────────────────────
# Each region has a list of countries shown in the dropdown.
# "Other" at the bottom of each region catches anything not listed.

REGIONS: dict[str, list[str]] = {
    "North America": [
        "Canada", "Mexico", "USA",
    ],
    "Latin America": [
        "Argentina", "Brazil", "Chile", "Colombia", "Ecuador",
        "Peru", "Uruguay", "Venezuela",
    ],
    "Europe": [
        "Austria", "Belgium", "Czech Republic", "Denmark", "Finland",
        "France", "Germany", "Greece", "Hungary", "Ireland",
        "Italy", "Netherlands", "Norway", "Poland", "Portugal",
        "Romania", "Russia", "Spain", "Sweden", "Switzerland",
        "Turkey", "Ukraine", "UK",
    ],
    "Middle East & Africa": [
        "Algeria", "Egypt", "Ethiopia", "Ghana", "Israel",
        "Jordan", "Kenya", "Morocco", "Nigeria", "Saudi Arabia",
        "South Africa", "Tanzania", "Tunisia", "UAE",
    ],
    "South Asia": [
        "Bangladesh", "India", "Nepal", "Pakistan", "Sri Lanka",
    ],
    "East & Southeast Asia": [
        "China", "Hong Kong", "Indonesia", "Japan", "Malaysia",
        "Philippines", "Singapore", "South Korea", "Taiwan", "Thailand",
        "Vietnam",
    ],
    "Oceania": [
        "Australia", "New Zealand",
    ],
}

# Flat ordered list for dropdowns: grouped by region with "Other" per region
def get_country_options() -> list[str]:
    options: list[str] = []
    for region, countries in REGIONS.items():
        for c in sorted(countries):
            options.append(c)
        options.append(f"Other ({region})")
    return options

ALL_COUNTRIES: list[str] = get_country_options()

# Reverse map: country → region
COUNTRY_TO_REGION: dict[str, str] = {}
for _region, _countries in REGIONS.items():
    for _c in _countries:
        COUNTRY_TO_REGION[_c] = _region
    COUNTRY_TO_REGION[f"Other ({_region})"] = _region

# Region-level context: typical market notes for Gemini
REGION_CONTEXT: dict[str, str] = {
    "North America":           "mature streaming market, high ARPU, premium plan penetration, English-language content dominant",
    "Latin America":           "price-sensitive market, growing mobile-first user base, strong telenovela & football interest",
    "Europe":                  "diverse languages & regulations, GDPR-aware, strong local content demand, mixed plan tiers",
    "Middle East & Africa":    "rapidly growing market, young demographics, Arabic & local content demand, mobile-heavy",
    "South Asia":              "price-sensitive, massive scale, Hindi/regional language content crucial, mobile-first",
    "East & Southeast Asia":   "anime & K-drama strong demand, mixed maturity, local competition intense (iQiyi, Viu, Disney+)",
    "Oceania":                 "small but high-ARPU market, English-language, similar taste profile to North America",
}

# ── Feature engineering constants ─────────────────────────────────
SUBSCRIPTIONS = ["Basic", "Premium", "Standard"]
GENRES        = ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"]

AGE_BINS   = [0, 13, 29, 46, 63, 200]
AGE_LABELS = ["Child", "Young", "Adult", "MidAge", "Senior"]

WATCH_BINS = [-0.01, 3.0, 6.0, 9.0, 1e9]
WATCH_LBL  = ["Low", "Medium", "High", "Very_High"]

ACT_BINS = [-0.01, 10, 30, 1e9]
ACT_LBL  = ["Active", "Medium", "Inactive"]

ENGAGEMENT_MEDIAN_REF = 0.25

# ── Plot layout ────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,22,31,0.8)",
    font=dict(family="DM Sans", color="#aaa", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Region color palette for charts
REGION_PALETTE = {
    "North America":           "#e50914",
    "Latin America":           "#f59e0b",
    "Europe":                  "#60a5fa",
    "Middle East & Africa":    "#a78bfa",
    "South Asia":              "#34d399",
    "East & Southeast Asia":   "#fb923c",
    "Oceania":                 "#f472b6",
}
