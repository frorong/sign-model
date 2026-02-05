#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
import yaml
import h5py

from models import SynthesisNetwork
from inference import Sampler


class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Synthesis Test")
        self.root.geometry("800x600")
        
        self.model = None
        self.sampler = None
        self.alphabet = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config_path = project_root / "configs" / "default.yaml"
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.load_alphabet()
        self.setup_ui()
        self.refresh_models()
    
    def load_alphabet(self):
        train_path = project_root / "data" / "train.h5"
        if train_path.exists():
            with h5py.File(train_path, 'r') as f:
                self.alphabet = f.attrs['alphabet']
        else:
            self.alphabet = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    
    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, width=50, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)
        
        ttk.Button(control_frame, text="Refresh", command=self.refresh_models).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Name (English):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.text_var = tk.StringVar(value="Seungje Lee")
        self.text_entry = ttk.Entry(control_frame, textvariable=self.text_var, width=40)
        self.text_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(control_frame, text="Generate", command=self.generate).grid(row=1, column=2, padx=5, pady=5)
        
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky=tk.W)
        
        ttk.Label(param_frame, text="Bias:").pack(side=tk.LEFT, padx=5)
        self.bias_var = tk.DoubleVar(value=0.5)
        bias_scale = ttk.Scale(param_frame, from_=0.0, to=2.0, variable=self.bias_var, orient=tk.HORIZONTAL, length=150)
        bias_scale.pack(side=tk.LEFT, padx=5)
        self.bias_label = ttk.Label(param_frame, text="0.5")
        self.bias_label.pack(side=tk.LEFT, padx=5)
        bias_scale.configure(command=lambda v: self.bias_label.configure(text=f"{float(v):.2f}"))
        
        ttk.Label(param_frame, text="Scale:").pack(side=tk.LEFT, padx=15)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_scale = ttk.Scale(param_frame, from_=0.5, to=3.0, variable=self.scale_var, orient=tk.HORIZONTAL, length=150)
        scale_scale.pack(side=tk.LEFT, padx=5)
        self.scale_label = ttk.Label(param_frame, text="1.0")
        self.scale_label.pack(side=tk.LEFT, padx=5)
        scale_scale.configure(command=lambda v: self.scale_label.configure(text=f"{float(v):.2f}"))
        
        progress_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        progress_frame.pack(fill=tk.X)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.progress_label = ttk.Label(progress_frame, text="0/0", width=10)
        self.progress_label.pack(side=tk.RIGHT, padx=5)
        
        canvas_frame = ttk.Frame(self.root, padding=10)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value=f"Device: {self.device}")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def refresh_models(self):
        checkpoint_dir = project_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        models = list(checkpoint_dir.glob("*.pt"))
        model_names = [m.name for m in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)]
        
        self.model_combo['values'] = model_names
        if model_names:
            self.model_combo.current(0)
            self.on_model_select(None)
    
    def on_model_select(self, event):
        model_name = self.model_var.get()
        if not model_name:
            return
        
        model_path = project_root / "checkpoints" / model_name
        self.status_var.set(f"Loading {model_name}...")
        self.root.update()
        
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
        self.status_var.set(f"Loaded: {model_name} | Alphabet: {len(self.alphabet)} chars | Device: {self.device}")
    
    def update_progress(self, current: int, total: int):
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.configure(text=f"{current}/{total}")
        self.root.update_idletasks()
    
    def generate(self):
        if self.sampler is None:
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        text = self.text_var.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text")
            return
        
        self.progress_var.set(0)
        self.progress_label.configure(text="0/700")
        self.status_var.set(f"Generating '{text}'...")
        self.root.update()
        
        bias = self.bias_var.get()
        scale = self.scale_var.get()
        
        strokes = self.sampler.generate(text, bias=bias, progress_callback=self.update_progress)
        self.draw_strokes(strokes, scale)
        
        self.progress_var.set(100)
        self.progress_label.configure(text=f"{len(strokes)}/{len(strokes)}")
        self.status_var.set(f"Generated '{text}' ({len(strokes)} strokes) | Device: {self.device}")
    
    def draw_strokes(self, strokes: np.ndarray, scale: float = 1.0):
        self.canvas.delete("all")
        
        coords = np.cumsum(strokes[:, :2], axis=0)
        coords = coords * scale * 50
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        
        offset_x = (canvas_width - (max_x - min_x)) / 2 - min_x
        offset_y = (canvas_height - (max_y - min_y)) / 2 - min_y
        
        coords[:, 0] += offset_x
        coords[:, 1] += offset_y
        
        pen_up = False
        current_stroke = []
        
        for i in range(len(coords)):
            if pen_up:
                if len(current_stroke) >= 2:
                    self.canvas.create_line(current_stroke, fill="black", width=2, smooth=True)
                current_stroke = [(coords[i, 0], coords[i, 1])]
                pen_up = False
            else:
                current_stroke.append((coords[i, 0], coords[i, 1]))
            
            if strokes[i, 2] > 0.5:
                pen_up = True
        
        if len(current_stroke) >= 2:
            self.canvas.create_line(current_stroke, fill="black", width=2, smooth=True)


def main():
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
