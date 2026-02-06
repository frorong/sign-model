#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import yaml
import h5py
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import json

from models import SynthesisNetwork
from inference import Sampler


class HandwritingGenerator:
    def __init__(self):
        self.model = None
        self.sampler = None
        self.alphabet = None
        self.current_model = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.load_alphabet()
    
    def load_alphabet(self):
        train_path = project_root / "data" / "train.h5"
        if train_path.exists():
            with h5py.File(train_path, 'r') as f:
                self.alphabet = f.attrs['alphabet']
        else:
            self.alphabet = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    
    def get_models(self):
        checkpoint_dir = project_root / "checkpoints"
        models = list(checkpoint_dir.glob("*.pt"))
        return [m.name for m in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)]
    
    def load_model(self, model_name):
        if not model_name:
            return False
        
        model_path = project_root / "checkpoints" / model_name
        
        self.model = SynthesisNetwork(
            alphabet_size=len(self.alphabet),
            hidden_size=self.config['model']['hidden_size'],
            num_mixtures=self.config['model']['num_mixtures'],
            num_attention_components=self.config['model']['num_attention_components']
        )
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        
        self.sampler = Sampler(self.model, self.alphabet, self.device)
        self.current_model = model_name
        return True
    
    def generate(self, text, bias=0.5, scale=1.0):
        if self.sampler is None:
            return None
        
        strokes = self.sampler.generate(text.strip(), bias=bias)
        return self.strokes_to_svg(strokes, scale)
    
    def strokes_to_svg(self, strokes, scale=1.0):
        coords = np.cumsum(strokes[:, :2], axis=0) * scale * 50
        
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        width = max_x - min_x + 40
        height = max_y - min_y + 40
        
        path_parts = [f"M {coords[0, 0] - min_x + 20:.2f} {coords[0, 1] - min_y + 20:.2f}"]
        pen_up = False
        
        for i in range(1, len(coords)):
            x = coords[i, 0] - min_x + 20
            y = coords[i, 1] - min_y + 20
            if pen_up:
                path_parts.append(f"M {x:.2f} {y:.2f}")
                pen_up = False
            else:
                path_parts.append(f"L {x:.2f} {y:.2f}")
            if strokes[i, 2] > 0.5:
                pen_up = True
        
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect width="100%" height="100%" fill="white"/>
  <path d="{' '.join(path_parts)}" fill="none" stroke="black" stroke-width="2"/>
</svg>'''


generator = HandwritingGenerator()


HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Signature Synthesis</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .control {{ margin: 15px 0; display: flex; align-items: center; gap: 10px; }}
        label {{ display: inline-block; width: 100px; font-weight: 500; }}
        select, input[type="text"] {{ padding: 10px; font-size: 14px; border: 1px solid #ddd; border-radius: 5px; }}
        button {{ padding: 10px 20px; font-size: 14px; cursor: pointer; background: #007AFF; color: white; border: none; border-radius: 5px; }}
        button:hover {{ background: #0056b3; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        #result {{ padding: 30px; border: 2px dashed #ddd; border-radius: 10px; min-height: 150px; background: white; display: flex; justify-content: center; align-items: center; }}
        #status {{ margin-top: 10px; color: #666; font-size: 13px; }}
        .slider-container {{ display: flex; align-items: center; gap: 10px; flex: 1; }}
        input[type="range"] {{ width: 200px; }}
        .style-section {{ border-top: 1px solid #eee; margin-top: 15px; padding-top: 15px; }}
        .style-section h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #666; }}
        .style-options {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .style-btn {{ padding: 8px 16px; border: 2px solid #ddd; background: white; border-radius: 20px; cursor: pointer; font-size: 13px; }}
        .style-btn.active {{ border-color: #007AFF; background: #f0f7ff; }}
        .style-btn:hover {{ border-color: #007AFF; }}
    </style>
</head>
<body>
    <h1>Signature Synthesis</h1>
    
    <div class="card">
        <div class="control">
            <label>Model:</label>
            <select id="model" style="flex: 1; max-width: 300px;">{model_options}</select>
            <button onclick="loadModel()">Load</button>
        </div>
        
        <div class="control">
            <label>Name:</label>
            <input type="text" id="text" value="Seungje Lee" style="flex: 1; max-width: 300px;">
            <button onclick="generate()" id="generateBtn">Generate Signature</button>
        </div>
        
        <div class="control slider-container">
            <label>Sharpness:</label>
            <input type="range" id="bias" min="0" max="3" step="0.1" value="1.0">
            <span id="biasValue">1.0</span>
        </div>
        
        <div class="control slider-container">
            <label>Size:</label>
            <input type="range" id="scale" min="0.5" max="3" step="0.1" value="1.0">
            <span id="scaleValue">1.0</span>
        </div>
        
        <div class="style-section">
            <h3>Signature Style (Stage 2 - Coming Soon)</h3>
            <div class="style-options">
                <button class="style-btn active" disabled>Default</button>
                <button class="style-btn" disabled>Formal</button>
                <button class="style-btn" disabled>Casual</button>
                <button class="style-btn" disabled>Artistic</button>
                <button class="style-btn" disabled>Custom...</button>
            </div>
        </div>
    </div>
    
    <div id="status">Device: {device} | Model: Not loaded</div>
    <div id="result">Generate a signature to see the result</div>
    
    <script>
        document.getElementById('bias').oninput = e => document.getElementById('biasValue').textContent = e.target.value;
        document.getElementById('scale').oninput = e => document.getElementById('scaleValue').textContent = e.target.value;
        
        function loadModel() {{
            const model = document.getElementById('model').value;
            document.getElementById('status').textContent = 'Loading model...';
            fetch('/load?model=' + encodeURIComponent(model))
                .then(r => r.json())
                .then(d => document.getElementById('status').textContent = d.status);
        }}
        
        function generate() {{
            const text = document.getElementById('text').value;
            const bias = document.getElementById('bias').value;
            const scale = document.getElementById('scale').value;
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.textContent = 'Generating...';
            document.getElementById('status').textContent = 'Generating signature...';
            fetch('/generate?text=' + encodeURIComponent(text) + '&bias=' + bias + '&scale=' + scale)
                .then(r => r.json())
                .then(d => {{
                    document.getElementById('result').innerHTML = d.svg || d.error;
                    document.getElementById('status').textContent = d.status;
                    btn.disabled = false;
                    btn.textContent = 'Generate Signature';
                }});
        }}
    </script>
</body>
</html>'''


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        if parsed.path == '/':
            models = generator.get_models()
            options = ''.join(f'<option value="{m}">{m}</option>' for m in models)
            html = HTML_TEMPLATE.format(model_options=options, device=generator.device)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        
        elif parsed.path == '/load':
            model = params.get('model', [''])[0]
            success = generator.load_model(model)
            status = f"Device: {generator.device} | Model: {model}" if success else "Failed to load"
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': status}).encode())
        
        elif parsed.path == '/generate':
            text = params.get('text', [''])[0]
            bias = float(params.get('bias', ['0.5'])[0])
            scale = float(params.get('scale', ['1.0'])[0])
            
            if generator.sampler is None:
                result = {'error': 'Model not loaded', 'status': 'Please load a model first'}
            else:
                svg = generator.generate(text, bias, scale)
                result = {'svg': svg, 'status': f"Generated '{text}' | Model: {generator.current_model}"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass


def main():
    port = 7860
    print(f"Starting server at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop")
    server = HTTPServer(('127.0.0.1', port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
