import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

try:
    from pickle import load as pickle_load
except Exception:
    pickle_load = None

try:
    import cvxopt as opt
    from cvxopt import solvers
except Exception:
    opt = None
    solvers = None

# ---------- Theming ----------
THEME = dbc.themes.FLATLY  # flat, modern palette
app = Dash(__name__, external_stylesheets=[THEME])
server = app.server

CARD_STYLE = {"borderRadius":"1rem", "boxShadow":"0 8px 24px rgba(0,0,0,0.08)"}
SECTION_PAD = {"padding":"1rem 1.25rem"}

# ---------- Data ----------
DATA_DIR = os.environ.get("DATA_DIR", ".")
INVESTORS_CSV = os.path.join(DATA_DIR, "InputData.csv")
ASSETS_CSV = os.path.join(DATA_DIR, "SP500Data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "finalized_model.sav")

def load_investors():
    if not os.path.exists(INVESTORS_CSV):
        cols = ["AGE07","NETWORTH07","INCOME07","EDCL07","MARRIED07","KIDS07","OCCAT107","RISK07"]
        df = pd.DataFrame([{ "AGE07":25, "NETWORTH07":10000, "INCOME07": 100000, "EDCL07":2,
                             "MARRIED07":1, "KIDS07":0, "OCCAT107":3, "RISK07":3 }], columns=cols)
        return df
    return pd.read_csv(INVESTORS_CSV, index_col=0)

def load_assets():
    if not os.path.exists(ASSETS_CSV):
        dates = pd.date_range("2018-01-01", periods=400, freq="B")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "AAPL": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "MSFT": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "GOOGL": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "META": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "GE": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "GS": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "MS": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
            "AIZ": 100 + np.cumsum(rng.normal(0, 1, len(dates))),
        }, index=dates)
        return df
    df = pd.read_csv(ASSETS_CSV, index_col=0)
    missing_fractions = df.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    df.drop(labels=drop_list, axis=1, inplace=True)
    df = df.fillna(method="ffill")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

investors = load_investors()
assets = load_assets()
options = [{"label": tic, "value": tic} for tic in assets.columns]

# ---------- Helpers ----------
def _project_to_simplex(v):
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def mean_variance_pg(mean_ret, cov, lam, iters=700, lr=0.05, w0=None):
    n = mean_ret.shape[0]
    w = np.ones(n)/n if w0 is None else _project_to_simplex(w0)
    for _ in range(iters):
        grad = mean_ret - 2.0 * lam * (cov @ w)
        w = w + lr * grad
        w = _project_to_simplex(w)
    return w

def predict_risk_tolerance(X_input):
    def _as_dataframe(values, feature_names=None):
        import pandas as pd
        if isinstance(values, dict):
            if feature_names is None:
                return pd.DataFrame([values])
            row = {k: values.get(k) for k in feature_names}
            return pd.DataFrame([row], columns=feature_names)
        guess_names = ["AGE07","EDCL07","MARRIED07","KIDS07","OCCAT107","INCOME07","RISK07","NETWORTH07"]
        if feature_names is None:
            return pd.DataFrame(values, columns=guess_names[:len(values[0])])
        row_map = {name: val for name, val in zip(guess_names, values[0])}
        row = [row_map.get(k, 0.0) for k in feature_names]
        return pd.DataFrame([row], columns=feature_names)

    if os.path.exists(MODEL_PATH) and pickle_load is not None:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle_load(f)
            names_expected = getattr(model, "feature_names_in_", None)
            X_df = _as_dataframe(X_input, names_expected)
            pred = np.asarray(model.predict(X_df), dtype=float)
            if pred.max() > 1.5:
                pred = np.clip(pred / 100.0, 0.0, 1.0)
            return float(np.clip(pred[0], 0.0, 1.0))
        except Exception:
            pass

    if isinstance(X_input, dict):
        age = float(X_input.get("AGE07", 30))
        risk4 = float(X_input.get("RISK07", 3))
    else:
        age = float(X_input[0][0])
        risk4 = float(X_input[0][6])
    base = (risk4 - 1) / 3.0
    age_penalty = max(0, (age - 25) / (70 - 25)) * 0.3
    return float(np.clip(base * (1 - 0.2) + (0.2) * (1 - age_penalty), 0.0, 1.0))

def get_asset_allocation(risk_tolerance_0_100, tickers):
    if not tickers:
        raise ValueError("Please select at least one asset.")
    selected = [t for t in tickers if t in assets.columns]
    if not selected:
        raise ValueError("Selected tickers not found in asset data.")
    R = assets.loc[:, selected].pct_change().dropna(axis=0)
    returns = R.T.values
    n = returns.shape[0]
    if n == 0:
        raise ValueError("Not enough data to compute returns.")
    cov = np.cov(returns) + 1e-6 * np.eye(n)
    mean_ret = returns.mean(axis=1)

    rt = float(risk_tolerance_0_100 or 50.0)
    rt01 = np.clip(rt / 100.0, 0.0, 1.0)
    lam = 0.1 + 4.0 * (1.0 - rt01)

    used_solver = "pg"
    w = None

    if opt is not None and solvers is not None:
        try:
            S = opt.matrix(cov)
            pbar = opt.matrix(mean_ret)
            G = -opt.matrix(np.eye(n)); h = opt.matrix(0.0, (n, 1))
            A = opt.matrix(1.0, (1, n)); b = opt.matrix(1.0)
            solvers.options["show_progress"] = False
            sol = solvers.qp(2.0 * lam * S, -pbar, G, h, A, b)
            w = np.array(sol["x"]).reshape(-1)
            used_solver = "cvxopt"
        except Exception:
            w = None
    if w is None:
        w = mean_variance_pg(mean_ret, cov, lam, iters=700, lr=0.05)
        used_solver = "pg"

    alloc = pd.DataFrame(w, index=selected, columns=["weight"])
    weights = w.reshape(-1)
    port_ret = (R.values @ weights)
    cum_value = 100 * (1 + pd.Series(port_ret, index=R.index)).cumprod()
    perf = pd.DataFrame({"Portfolio": cum_value})
    alloc.attrs["solver"] = used_solver
    return alloc, perf

# ---------- UI ----------
def section_header(title, subtitle=None, icon="bar-chart-fill"):
    return html.Div([
        html.Div([
            html.H3(title, className="mb-0"),
            html.P(subtitle, className="text-muted mb-0") if subtitle else html.Span()
        ], style={"padding":"0.5rem 0"})
    ])

# Left controls card
controls = dbc.Card([
    dbc.CardHeader("Step 1 · Investor Characteristics", className="fw-bold"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Age"), dcc.Slider(id="Age", min=25, max=70, step=1,
                marks={25:"25",35:"35",45:"45",55:"55",70:"70"}, value=25)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Net Worth"), dcc.Slider(id="Nwcat", min=-1000000, max=3000000, step=10000,
                marks={-1000000:"-$1M",0:"0",500000:"$500K",1000000:"$1M",2000000:"$2M"}, value=10000)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Income"), dcc.Slider(id="Inccl", min=-1000000, max=3000000, step=10000,
                marks={-1000000:"-$1M",0:"0",500000:"$500K",1000000:"$1M",2000000:"$2M"}, value=100000)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Education (1–4)"), dcc.Slider(id="Edu", min=1, max=4, step=1,
                marks={1:"1",2:"2",3:"3",4:"4"}, value=2)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Married (1/2)"), dcc.Slider(id="Married", min=1, max=2, step=1,
                marks={1:"1",2:"2"}, value=1)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Kids"), dcc.Slider(id="Kids", min=0, max=6, step=1,
                marks={i:str(i) for i in range(0,7)}, value=0)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Occupation (1–4)"), dcc.Slider(id="Occ", min=1, max=4, step=1,
                marks={1:"1",2:"2",3:"3",4:"4"}, value=3)], width=12)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([html.Label("Willingness to take Risk (1–4)"),
                     dcc.Slider(id="Risk", min=1, max=4, step=1, marks={1:"1",2:"2",3:"3",4:"4"}, value=3)], width=12)
        ], className="mb-3"),
        dbc.Button("Calculate Risk Tolerance", id="investor_char_button", color="primary", className="w-100")
    ])
], style=CARD_STYLE)

# Right config + charts
right_top = dbc.Card([
    dbc.CardHeader("Step 2 · Asset Selection", className="fw-bold"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Risk Tolerance (0–100)"),
                dcc.Input(id="risk-tolerance-text", type="number", value=50, debounce=True, className="form-control")
            ], md=4),
            dbc.Col([
                html.Label("Select assets"),
                dcc.Dropdown(id="ticker_symbol", options=options,
                             value=["GOOGL","MSFT","GS","MS","GE","AIZ"],
                             multi=True, persistence=True, persistence_type="memory")
            ], md=8)
        ], className="gy-2"),
        dbc.Row([
            dbc.Col(dbc.Button("Submit", id="submit-asset_alloc_button", color="dark", className="mt-2"), width="auto")
        ])
    ])
], style=CARD_STYLE)

charts = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="Asset-Allocation", config={"displayModeBar": True}), type="dot"), md=6),
            dbc.Col(dcc.Loading(dcc.Graph(id="Performance", config={"displayModeBar": True}), type="dot"), md=6)
        ], className="gy-3")
    ])
], style=CARD_STYLE)

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Robo Advisor", className="ms-2 fw-bold"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
        ], className="ms-auto", navbar=True)
    ]),
    color="darkblue", dark=True, sticky="top"
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        html.Div(style={"height":"1rem"}),
        dbc.Row([
            dbc.Col(controls, md=4, className="mb-4"),
            dbc.Col([right_top, html.Div(style={"height":"1rem"}), charts], md=8)
        ]),
        html.Div(style={"height":"1rem"}),
        html.Footer(html.Small("© Robo Advisor Demo — built with Dash & Plotly"), className="text-center text-muted mb-4")
    ], fluid=True, className="py-2")
])

# ---------- Callbacks ----------
@app.callback(
    Output("risk-tolerance-text", "value"),
    Input("investor_char_button", "n_clicks"),
    State("Age", "value"),
    State("Nwcat", "value"),
    State("Inccl", "value"),
    State("Risk", "value"),
    State("Edu", "value"),
    State("Married", "value"),
    State("Kids", "value"),
    State("Occ", "value"),
    prevent_initial_call=True,
)
def update_risk_tolerance(n_clicks, Age, Nwcat, Inccl, Risk, Edu, Married, Kids, Occ):
    X_input = [[Age, Edu, Married, Kids, Occ, Inccl, Risk, Nwcat]]
    rt01 = predict_risk_tolerance(X_input)
    return round(float(rt01 * 100), 2)

@app.callback(
    Output("Asset-Allocation", "figure"),
    Output("Performance", "figure"),
    Input("submit-asset_alloc_button", "n_clicks"),
    State("risk-tolerance-text", "value"),
    State("ticker_symbol", "value"),
    prevent_initial_call=True,
)
def update_asset_allocation_chart(n_clicks, risk_tolerance, stock_ticker):
    try:
        alloc, perf = get_asset_allocation(risk_tolerance, stock_ticker)
    except Exception as e:
        empty1 = go.Figure()
        empty1.update_layout(template="plotly_white",
                             title=f"Asset allocation - Error: {str(e)}",
                             margin=dict(l=40,r=20,t=60,b=40))
        empty2 = go.Figure()
        empty2.update_layout(template="plotly_white",
                             title="Portfolio value of $100 investment",
                             margin=dict(l=40,r=20,t=60,b=40))
        return empty1, empty2

    fig_alloc = go.Figure()
    fig_alloc.add_bar(x=alloc.index, y=alloc["weight"])
    fig_alloc.update_layout(
        template="plotly_white",
        title=f"Asset allocation - Mean-Variance Allocation ({alloc.attrs.get('solver','pg')})",
        xaxis_title="Ticker", yaxis_title="Weight",
        yaxis_tickformat=".0%",
        #margin=dict(l=40,r=20,t=60,b=40),
        hovermode="x unified"
    )

    fig_perf = go.Figure()
    fig_perf.add_scatter(x=perf.index, y=perf["Portfolio"], mode="lines", name="Portfolio")
    fig_perf.update_layout(
        template="plotly_white",
        title="Portfolio value of $100 investment",
        xaxis_title="Date", yaxis_title="Value ($)",
        #margin=dict(l=40,r=20,t=60,b=40),
        hovermode="x unified"
    )

    return fig_alloc, fig_perf

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
