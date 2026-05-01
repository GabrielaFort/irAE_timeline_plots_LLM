from datetime import date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = {
    "treatment": "#1f77b4",  # blue
    "irae": "#d62728",       # red
}

def parse_dates(s):
    """Parse a date string in the format YYYY-MM-DD, YYYY-MM, or YYYY."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"none", "null", "na", "n/a"}:
        return None
    if len(s) == 4:
        return date(int(s), 1, 1)
    if len(s) == 7:
        year, month = map(int, s.split("-"))
        return date(year, month, 1)
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

def make_plot(parsed_events, file_name):
    """Build a timeline plot from parsed LLM events.
    `parsed_events` items should include: 
    - condition_type: treatment | irae
    - condition: str
    - start_date: YYYY-MM-DD | YYYY-MM | YYYY
    - end_date: same format or empty/null
    """
    rows = []
    for e in parsed_events:
        # Extract all fields
        ctype = str(e.get("condition_type", "")).strip().lower()
        condition = e.get("condition")
        start_raw = e.get("start_date")
        end_raw = e.get("end_date")

        # Skip the row if any requitred fields are empty
        if not ctype or not condition or not start_raw:
            continue

        start = parse_dates(start_raw)
        end = parse_dates(end_raw) if end_raw else None

        if start is None:
            continue

        # If no end date given, treat as a point event at the start date.
        is_point = end is None

        # If no end date, set end date to same as start date
        finish = start if is_point else end

        # If end date is before start date, set end date to same as start date
        if finish is not None and finish < start:
            finish = start
            is_point = True
        
        if finish == start:
            is_point = True

        rows.append(
            {
                "type": ctype,
                "condition": condition,
                "label": f"{condition}",
                "start": start,
                "end": finish,
                "is_point": is_point,
                "start_raw": start_raw,
                "end_raw": end_raw if end_raw else None,
            }
        )

    # Empty if no events
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=f"Timeline: {file_name}")
        return fig
    
    # Convert results to df for plotting
    df = pd.DataFrame(rows)

    # Sort lines top to bottom by start date
    order = (
        df.groupby("label")["start"]
        .min()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Make figure
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="label",
        color="type",
        color_discrete_map=COLORS,
        category_orders={"label": order},
        hover_data={
            "condition": True,
            "type": True,
            "start_raw": True,
            "end_raw": True,
            "start": False,
            "end": False,
            "is_point": False,
            "label": False,
        },
        title=f"Treatment and irAE Timeline: {file_name}",
    )

    # Add dots to mark point events
    points = df[df["is_point"]]
    if not points.empty:
        for ctype, subset in points.groupby("type"):
            fig.add_trace(
                go.Scatter(
                    x=subset["start"],
                    y=subset["label"],
                    mode="markers",
                    marker={"size": 10, "color": COLORS.get(ctype, "#555")},
                    name=f"{ctype} (point)",
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Type: " + ctype + "<br>"
                        "Date: %{x|%Y-%m-%d}<extra></extra>"
                    ),
                )
            )

    fig.update_yaxes(title="Event", autorange="reversed")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        bargap=0.35,
        template="plotly_white",
        legend_title_text="Event Type",
        height=max(420, 70 + 38 * df["label"].nunique()),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig
