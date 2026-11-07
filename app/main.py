# app/main.py
import os
import threading
from flask import Flask, render_template_string, request, jsonify
from app import config
from app.backend import LLMBackend
from app.pipeline import AnalysisPipeline
from app.ram_monitor import RamMonitor

app = Flask(__name__)
backend = None
pipeline = None

# Initialize and start the background RAM monitor daemon
ram_monitor = RamMonitor().start()

# Background initialization of model to keep startup non-blocking
def init_model():
    global backend, pipeline
    if backend is None:
        backend = LLMBackend()
        pipeline = AnalysisPipeline(backend)
        ram_monitor.mark("model_loaded")

threading.Thread(target=init_model, daemon=True).start()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Local AI Security Log Analyzer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; color: #333; }
        h1 { color: #2c3e50; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        pre { background: #f1f3f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
        .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        .Critical { background: #dc3545; color: white; }
        .High { background: #fd7e14; color: white; }
        .Medium { background: #ffc107; color: black; }
        .Low { background: #28a745; color: white; }
    </style>
</head>
<body>
    <h1>🔒 Local AI Security Log Analyzer</h1>
    <div class="card">
        <h3>System Status</h3>
        <p>RAM Usage: <span id="ram-usage">Loading...</span> | Model: Qwen3.5-0.8B (fp32 local)</p>
    </div>
    
    <div class="card">
        <h3>Upload Log File</h3>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="logfile" accept=".log,.txt" required /><br><br>
            <button type="submit" id="submit-btn">Start Analysis</button>
        </form>
        <p id="status-msg"></p>
    </div>

    <div class="card" id="results-card" style="display:none;">
        <h3>Final Security Report</h3>
        <div id="findings-container"></div>
    </div>

<script>
async function updateRAM() {
    const res = await fetch('/api/ram');
    const data = await res.json();
    document.getElementById('ram-usage').innerText = data.rss_mb + " MB / 4096 MB (" + data.percent + "%)";
}
setInterval(updateRAM, 3000);
updateRAM();

document.getElementById('upload-form').onsubmit = async (e) => {
    e.preventDefault();
    const btn = document.getElementById('submit-btn');
    const statusMsg = document.getElementById('status-msg');
    btn.disabled = true;
    statusMsg.innerText = "Analyzing log chunks locally (this may take a minute)...";

    const formData = new FormData(e.target);
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await res.json();

    btn.disabled = false;
    if (data.error) {
        statusMsg.innerText = "Error: " + data.error;
        return;
    }

    statusMsg.innerText = "Analysis complete! Processed " + data.total_chunks_processed + " chunks.";
    
    const container = document.getElementById('findings-container');
    container.innerHTML = "";
    if (data.findings.length === 0) {
        container.innerHTML = "<p>No security findings detected.</p>";
    } else {
        data.findings.forEach(f => {
            container.innerHTML += `
                <div style="border-bottom: 1px solid #eee; padding: 10px 0;">
                    <span class="badge ${f.severity}">${f.severity}</span>
                    <strong>${f.attack_type}</strong> (Target: ${f.target_id})<br>
                    <small>Reason: ${f.reason}</small><br>
                    <small>Action: ${f.action}</small>
                </div>
            `;
        });
    }
    document.getElementById('results-card').style.display = "block";
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/ram")
def ram_api():
    snap = ram_monitor.snapshot()
    return jsonify({
        "rss_mb": snap["rss_mb"],
        "percent": snap["used_pct_of_limit"]
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    if pipeline is None:
        return jsonify({"error": "Model is still loading in background. Please try again in a few seconds."}), 503
    
    if "logfile" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["logfile"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_path = os.path.join(config.PROJECT_ROOT, "samples", "uploaded_target.log")
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)
    file.save(upload_path)

    try:
        report = pipeline.run(upload_path)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)