import pandas as pd
import numpy as np
import joblib

from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go


df = pd.read_csv("industrial milling tool life dataset.csv")
rul_model = joblib.load("RUL_pred.pkl")

#load from pkl file
r2_score    = float(np.mean(np.ravel(joblib.load("r2.pkl"))))
rmse_score  = float(np.mean(np.ravel(joblib.load("rmse.pkl"))))
mae_score   = float(np.mean(np.ravel(joblib.load("mae.pkl"))))
kfold_score = float(np.mean(np.ravel(joblib.load("kfold.pkl"))))

# 3x2 scenarios - 3 tools , 2 coolant conditions
scenarios = [
    ("Aluminum", "ON"), ("Aluminum", "OFF"),
    ("Steel", "ON"), ("Steel", "OFF"),
    ("Titanium", "ON"), ("Titanium", "OFF"),
]

KPI_SCENARIO = ("Steel", "ON")

# def machine state
machine_state = {}
for mat, coolant in scenarios:
    machine_state[(mat, coolant)] = {
        "Spindle_Speed": None,
        "Feed_Rate": None,
        "Cutting_Depth": None,
        "Spindle_Load": None,
        "Vibration_X": None,
        "Vibration_Y": None,
        "Acoustic_Signal": None,
        "Tool_Wear_Level": None,
        "Material_Type": mat,
        "Coolant_Condition": coolant,
    }

rul_history = {key: [] for key in machine_state}
vopt_latest = {key: None for key in machine_state}
failure_latest = {key: None for key in machine_state}
time_steps = []


app = Dash(__name__)
app.title = "SIMULATED VIRTUAL FACTORY SHOWING PREDICTED REMAINING USEFUL TIME OF DIFFERENT TOOLS"

# metrics rep in card form
metric_card_style = {
    "padding": "18px",
    "borderRadius": "14px",
    "background": "#08154D",  
    "minWidth": "160px",
    "textAlign": "center",
    "color": "white",
    "boxShadow": "0 6px 14px rgba(0,0,0,0.35)"
}

# layout
app.layout = html.Div(
    style={"backgroundColor": "#020617", "color": "white", "padding": "25px"},
    children=[

        html.H2(
            "SIMULATED VIRTUAL FACTORY SHOWING PREDICTED REMAINING USEFUL TIME OF DIFFERENT TOOLS",
            style={"textAlign": "center", "marginBottom": "25px"}
        ),

        # manual input
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "12px",
                "marginBottom": "20px"
            },
            children=[
                dcc.Input(id="manual-speed", type="number", placeholder="Spindle Speed"),
                dcc.Input(id="manual-feed", type="number", placeholder="Feed Rate"),
                dcc.Input(id="manual-depth", type="number", placeholder="Cutting Depth"),
                html.Button("APPLY", id="apply-btn", n_clicks=0)
            ]
        ),
        

        # rep of key inputs 
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "40px",
                "marginBottom": "25px"
            },
            children=[
                html.Div(id="kpi-speed", style={"textAlign": "center"}),
                html.Div(id="kpi-load", style={"textAlign": "center"}),
                html.Div(id="kpi-wear", style={"textAlign": "center"}),
                html.Div(id="kpi-vibx", style={"textAlign": "center"}),
                html.Div(id="kpi-viby", style={"textAlign": "center"}),
                html.Div(id="kpi-acoustic", style={"textAlign": "center"}),
            ]
        ),

        # 6-graphs (3x2-3 tools + 2 coolant condition)
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "20px",
                "marginBottom": "45px"
            },
            children=[
                dcc.Graph(id="g-al-on"),
                dcc.Graph(id="g-al-off"),
                dcc.Graph(id="g-steel-on"),
                dcc.Graph(id="g-steel-off"),
                dcc.Graph(id="g-ti-on"),
                dcc.Graph(id="g-ti-off"),
            ]
        ),

        # performace metrics (r2, rmse, mae, kfold)
        html.H3(
            "MODEL PERFORMANCE METRICS",
            style={"textAlign": "center", "marginBottom": "15px"}
        ),

        html.Div(
            style={
                "display": "flex",
                "justifyContent": "center",
                "gap": "25px"
            },
            children=[
                html.Div([html.H4("R²"), html.H3(f"{r2_score:.3f}")], style=metric_card_style),
                html.Div([html.H4("RMSE"), html.H3(f"{rmse_score:.3f}")], style=metric_card_style),
                html.Div([html.H4("MAE"), html.H3(f"{mae_score:.3f}")], style=metric_card_style),
                html.Div([html.H4("K-FOLD"), html.H3(f"{kfold_score:.3f}")], style=metric_card_style),
            ]
        ),

        dcc.Interval(id="interval", interval=1000, n_intervals=0),

        html.Div(id="alert_txt"),
        dcc.Interval(id="alert_interval",interval=3000)
    ]
)

# figure rep
def make_fig(time, rul, vopt_value, failure_flag, title):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=rul,
        mode="lines",
        line=dict(width=2)
    ))

    failure_text = "YES" if failure_flag == 1 else "NO"

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showline=True, linecolor="white"),
        yaxis=dict(showgrid=False, showline=True, linecolor="white"),
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        height=300,
        font=dict(color="white"),
        annotations=[
            dict(
                x=0.98,
                y=0.95,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="right",
                bgcolor="rgba(0,0,0,0.45)",
                bordercolor="white",
                borderwidth=1,
                text=(
                    f"<b>RUL:</b> {rul[-1]:.2f}<br>"
                    f"<b>OPTIMUM VELOCITY:</b> {vopt_value:.1f} m/min<br>"
                    f"<b>FAILURE</b> {failure_text}"
                )
            )
        ]
    )

    return fig

# callback
@app.callback(
    Output("g-al-on", "figure"),
    Output("g-al-off", "figure"),
    Output("g-steel-on", "figure"),
    Output("g-steel-off", "figure"),
    Output("g-ti-on", "figure"),
    Output("g-ti-off", "figure"),
    Output("kpi-speed", "children"),
    Output("kpi-load", "children"),
    Output("kpi-wear", "children"),
    Output("kpi-vibx", "children"),
    Output("kpi-viby", "children"),
    Output("kpi-acoustic", "children"),
    Output("alert_txt","children"),
    Input("alert_interval","n_intervals"),
    Input("interval", "n_intervals"),
    Input("apply-btn","n_clicks"),
    Input("manual-speed","value"),
    Input("manual-feed","value"),
    Input("manual-depth","value")
)

def update_dashboard(alert_n, n,
                     apply_click,
                     manual_speed,
                     manual_feed,
                     manual_depth):

    if n == 0:
        time_steps.clear()
        for key, state in machine_state.items():
            state["Spindle_Speed"] = np.random.uniform(2200, 2800)
            state["Feed_Rate"] = np.random.uniform(150, 250)
            state["Cutting_Depth"] = np.random.uniform(0.8, 1.5)
            state["Tool_Wear_Level"] = 0.05
            rul_history[key].clear()
            vopt_latest[key] = None
            failure_latest[key] = 0

    for key, state in machine_state.items():
        # Manual override (only if button clicked)
        if apply_click and apply_click > 0:
            if manual_speed is not None:
                state["Spindle_Speed"] = manual_speed
            if manual_feed is not None:
                state["Feed_Rate"] = manual_feed
            if manual_depth is not None:
                state["Cutting_Depth"] = manual_depth

        state["Spindle_Speed"] += np.random.normal(
            0, 0.0025 * state["Spindle_Speed"]
        )

        base_wear = (
            0.00015
            * (state["Spindle_Speed"] / 2600)
            * (state["Feed_Rate"] / 200)
            * (state["Cutting_Depth"] / 1.2)
        )

        material_factor = {
            "Aluminum": 0.8,
            "Steel": 1.0,
            "Titanium": 1.4
        }[state["Material_Type"]]

        base_wear *= material_factor

        if state["Coolant_Condition"] == "OFF":
            base_wear *= 1.3

        if state["Tool_Wear_Level"] > 0.6:
            base_wear *= 1.6

        if np.random.rand() < 0.01:
            base_wear += np.random.uniform(0.005, 0.02)

        state["Tool_Wear_Level"] = min(
            state["Tool_Wear_Level"] + base_wear,
            1.0
        )

        state["Spindle_Load"] = (
            30
            + 25 * state["Cutting_Depth"]
            + 45 * state["Tool_Wear_Level"]
            + np.random.normal(0, 2)
        )

        state["Vibration_X"] = np.random.normal(
            0.35 + 1.2 * state["Tool_Wear_Level"], 0.05
        )
        state["Vibration_Y"] = np.random.normal(
            0.35 + 1.2 * state["Tool_Wear_Level"], 0.05
        )
        state["Acoustic_Signal"] = np.random.normal(
            50 + 40 * state["Tool_Wear_Level"], 2.5
        )

        mat_enc = {"Aluminum": 0, "Steel": 1, "Titanium": 2}[state["Material_Type"]]
        cool_enc = 1 if state["Coolant_Condition"] == "ON" else 0

        X = np.array([[
            state["Spindle_Speed"], state["Feed_Rate"], state["Cutting_Depth"],
            state["Vibration_X"], state["Vibration_Y"], state["Acoustic_Signal"],
            state["Spindle_Load"], state["Tool_Wear_Level"], mat_enc, cool_enc
        ]])

        pred = np.ravel(rul_model.predict(X))
        rul = float(pred[0])
        failure_flag = int(pred[1])

        if rul_history[key]:
            rul = min(rul, rul_history[key][-1])
        rul = max(rul, 0)

        rul_history[key].append(rul)
        failure_latest[key] = failure_flag

        if mat_enc == 0:
            vopt_latest[key] = 600 / (rul ** 0.35)
        elif mat_enc == 1:
            vopt_latest[key] = 250 / (rul ** 0.25)
        else:
            vopt_latest[key] = 100 / (rul ** 0.20)

    time_steps.append(len(time_steps))

    figs = {
        k: make_fig(
            time_steps,
            rul_history[k],
            vopt_latest[k],
            failure_latest[k],
            f"{k[0]} | Coolant {k[1]}"
        )
        for k in rul_history
    }

    state = machine_state[KPI_SCENARIO]

    health_index = (
        0.30 * (1 / (1 + rul)) +
        0.25 * (1 - state["Tool_Wear_Level"]) +
        0.15 * (1 / (1 + state["Vibration_X"])) +
        0.10 * (1 / (1 + state["Vibration_Y"])) +
        0.10 * (1 / (1 + state["Spindle_Speed"] / 1000)) +
        0.10 * (1 / (1 + state["Acoustic_Signal"] / 50))
    )

    health_index = max(0, min(1, health_index))
    wear_pct = 100 * (1 - health_index)

    flash = (alert_n or 0) % 2 == 0


    if wear_pct < 65:
        alert_msg = "NORMAL: TOOL CONDITION HEALTHY"
        alert_style = {
            "color": "#22c55e",
            "backgroundColor": "#052e16",
            "border": "2px solid #22c55e",
            "padding": "14px",
            "borderRadius": "12px",
            "textAlign": "center",
            "fontWeight": "bold"
        }

    elif wear_pct < 75:
        alert_msg = " WARNING: TOOL IS APPROACHING TOWARDS FAILURE"
        alert_style = {
            "color": "#f97316",
            "backgroundColor": "#431407",
            "border": "2px solid #f97316",
            "padding": "14px",
            "borderRadius": "12px",
            "textAlign": "center",
            "fontWeight": "bold"
        }

    else:
        alert_msg = "CRITICAL: TOOL FAILURE IMMINENT"
        alert_style = {
            "color": "white",
            "backgroundColor": "#dc2626" if flash else "#020617",
            "border": "3px solid red",
            "boxShadow": "0 0 20px red" if flash else "none",
            "padding": "16px",
            "borderRadius": "14px",
            "textAlign": "center",
            "fontWeight": "bold"
        }
    # ==================================================================

    return (
        figs[("Aluminum","ON")],
        figs[("Aluminum","OFF")],
        figs[("Steel","ON")],
        figs[("Steel","OFF")],
        figs[("Titanium","ON")],
        figs[("Titanium","OFF")],
        html.H4(f"SPINDLE SPEED: {int(state['Spindle_Speed'])} RPM"),
        html.H4(f"SPINDLE LOAD: {state['Spindle_Load']:.1f} %"),
        html.H4(f"TOOL WEAR: {state['Tool_Wear_Level']*100:.1f} %"),
        html.H4(f"VIBRATION X: {state['Vibration_X']:.3f} g"),
        html.H4(f"VIBRATION Y: {state['Vibration_Y']:.3f} g"),
        html.H4(f"ACOUSTIC SIGNAL: {state['Acoustic_Signal']:.1f} dB"),
        html.H3(alert_msg, style=alert_style)
    )




if __name__ == "__main__":
    app.run(debug=True) 