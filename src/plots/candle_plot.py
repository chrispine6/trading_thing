import os
import uuid
import tempfile
from pathlib import Path
import pandas as pd
from flask import Flask, render_template_string, request, session
from werkzeug.utils import secure_filename
import plotly.graph_objs as go

app = Flask(__name__)
# Add session secret key and upload folder config
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_for_local")
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), "trading_thing_uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Candle/Line Chart Viewer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        html, body { width: 100vw; height: 100vh; margin: 0; padding: 0; background: #181818; color: #fff; font-family: sans-serif; overflow-x: hidden; }
        .container { width: 100%; height: 100vh; margin: 0; padding: 0; }
        label, select, input { color: #fff; background: #222; border: none; padding: 0.5em; }
        .chart { width: 100%; height: 80vh; background: #181818; }
        form { margin-bottom: 0.5em; }
        h2, h3 { margin-left: 1em; }
        .error { color: red; margin: 1em; }
        .current-file { margin-left: 1em; color: #9f9; }
    </style>
</head>
<body>
<div class="container">
    {% if filename %}
        <h2>{{ filename }}</h2>
    {% endif %}
    <form method="POST" enctype="multipart/form-data">
        <!-- make file optional (remove required) -->
        <label>Upload CSV: <input type="file" name="file"></label>
        <label>Chart Type:
            <select name="chart_type">
                <option value="candle" {% if chart_type == 'candle' %}selected{% endif %}>Candle</option>
                <option value="line" {% if chart_type == 'line' %}selected{% endif %}>Line</option>
            </select>
        </label>
        <label>From: <input type="date" name="start_date" value="{{ start_date }}"></label>
        <label>To: <input type="date" name="end_date" value="{{ end_date }}"></label>
        <button type="submit">Show Chart</button>
        <!-- allow clearing the stored upload -->
        <button type="submit" name="clear" value="1">Clear uploaded file</button>
    </form>

    <!-- show the current uploaded filename so user knows what is being used -->
    {% if filename %}
        <div class="current-file">Current uploaded: {{ filename }}</div>
    {% endif %}

    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if plot_div %}
        <h3>{{ filename }}</h3>
        <div class="chart" style="width:100%;height:80vh;">{{ plot_div|safe }}</div>
    {% endif %}
</div>
</body>
</html>
"""

def make_plot(df, filename, chart_type):
    # Normalize column names for robust access
    df.columns = [col.strip().lower() for col in df.columns]
    # Map for standard OHLC names
    col_map = {}
    for std in ['open', 'high', 'low', 'close']:
        for col in df.columns:
            if col == std:
                col_map[std] = col
                break
    # Check for required columns
    required = ['open', 'high', 'low', 'close']
    if not all(k in col_map for k in required):
        raise ValueError("CSV must contain columns: Open, High, Low, Close (case-insensitive)")

    if chart_type == "line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_map['close']],
            mode='lines',
            line=dict(color='lime'),
            name='Close'
        ))
        y_min = df[col_map['close']].min()
        y_max = df[col_map['close']].max()
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df[col_map['open']],
            high=df[col_map['high']],
            low=df[col_map['low']],
            close=df[col_map['close']],
            increasing=dict(line=dict(color='lime'), fillcolor='lime'),
            decreasing=dict(line=dict(color='red'), fillcolor='red'),
            name='Candles'
        )])
        y_min = df[col_map['low']].min()
        y_max = df[col_map['high']].max()

    # Expand y-range to use even more vertical space (add 40% padding)
    y_range = y_max - y_min
    pad = y_range * 0.4 if y_range > 0 else 1
    yaxis_range = [y_min - pad, y_max + pad]

    fig.update_layout(
        plot_bgcolor='#181818',
        paper_bgcolor='#181818',
        font_color='white',
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333', range=yaxis_range, automargin=True, constrain='range'),
        title=filename,
        hovermode='x unified',
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_layout(width=None, height=None)
    return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True}, full_html=False)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_div = None
    # initialize filename from session so the UI shows the saved name on GET
    filename = session.get('uploaded_orig_name', "")
    chart_type = request.form.get("chart_type", "candle")
    start_date = request.form.get("start_date", "")
    end_date = request.form.get("end_date", "")
    error = None

    saved_path = None

    if request.method == "POST":
        # If user requested to clear the saved upload, remove it from session/disk
        if request.form.get("clear"):
            saved = session.pop('uploaded_path', None)
            session.pop('uploaded_orig_name', None)
            if saved and os.path.exists(saved):
                try:
                    os.remove(saved)
                except Exception:
                    pass
            filename = ""
            saved_path = None
            # If they only cleared, don't attempt to load a CSV this request
            # but continue so UI updates to reflect cleared state.

        # If a new file is uploaded, save it and store path in session
        file = request.files.get("file")
        if file and file.filename:
            try:
                original_name = file.filename
                uid = uuid.uuid4().hex
                safe_name = secure_filename(original_name) or "uploaded.csv"
                saved_name = f"{uid}_{safe_name}"
                saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
                file.save(saved_path)
                session['uploaded_path'] = saved_path
                session['uploaded_orig_name'] = original_name
                filename = original_name
            except Exception as e:
                error = f"Error saving uploaded file: {e}"
        else:
            # No new file in request; try to use previously uploaded file from session
            # (this allows changing date range without re-selecting the file)
            if not request.form.get("clear"):
                saved_path = session.get('uploaded_path')
                filename = session.get('uploaded_orig_name', filename)

        # If we have a file path (new or from session), proceed to load and filter
        if saved_path and os.path.exists(saved_path):
            try:
                # Read header to find the date column (case-insensitive)
                header = pd.read_csv(saved_path, nrows=0)
                date_col = None
                for col in header.columns:
                    if col.strip().lower() == "date":
                        date_col = col
                        break
                if date_col is None:
                    date_col = header.columns[0]
                df = pd.read_csv(saved_path)
                df.columns = [col.strip() for col in df.columns]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                df = df.sort_index()
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                if df.empty:
                    error = "No data available for the selected date range."
                else:
                    plot_div = make_plot(df, filename or os.path.basename(saved_path), chart_type)
            except Exception as e:
                error = f"Error loading CSV: {str(e)}"
        else:
            # If the user didn't clear and no saved file exists and none uploaded, prompt for upload
            if not error and not request.form.get("clear"):
                error = "Please upload a CSV file."

    return render_template_string(
        TEMPLATE,
        plot_div=plot_div,
        filename=filename,
        chart_type=chart_type,
        start_date=start_date,
        end_date=end_date,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
