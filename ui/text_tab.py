import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TextAnalysisTab:
    """Simplified text analysis tab UI component"""
    
    def __init__(self, parent, detector, status_var):
        self.parent = parent
        self.detector = detector
        self.status_var = status_var
        self.fig = None
        self.canvas = None
        
        # Basic emotion colors
        self.emotion_colors = {
            'happy': '#FFD700',  
            'joy': '#FFD700',    
            'sad': '#4169E1',    
            'angry': '#FF4500',  
            'fear': '#800080',   
            'disgust': '#006400',
            'surprise': '#FF69B4',
            'neutral': '#A9A9A9'
        }
        self.default_color = '#1E90FF'
        
        self.setup_ui()
    
    def setup_ui(self):
        # Text input section
        ttk.Label(self.parent, text="Enter text to analyze:").pack(anchor=tk.W, pady=(0, 5))

        self.text_input = tk.Text(self.parent, height=4, width=50, wrap=tk.WORD)
        self.text_input.pack(fill=tk.X, pady=5)

        # Sample text buttons
        sample_frame = ttk.Frame(self.parent)
        sample_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sample_frame, text="Examples:").pack(side=tk.LEFT, padx=(0, 5))

        samples = [
            "I am happy today!",
            "This makes me angry!",
            "I'm feeling sad",
            "That's surprising!"
        ]

        for sample in samples:
            btn = ttk.Button(sample_frame, text=sample,
                            command=lambda s=sample: self.set_sample_text(s))
            btn.pack(side=tk.LEFT, padx=2)

        # Analyze button
        ttk.Button(self.parent, text="Analyze", command=self.analyze_text).pack(pady=10)

        # Results section
        result_frame = ttk.Frame(self.parent)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Detected emotion label
        emotion_frame = ttk.Frame(result_frame)
        emotion_frame.pack(fill=tk.X, pady=5)

        ttk.Label(emotion_frame, text="Detected emotion:").pack(side=tk.LEFT, padx=(0, 10))
        self.emotion_label = ttk.Label(emotion_frame, text="", font=('Arial', 14, 'bold'))
        self.emotion_label.pack(side=tk.LEFT)

        # Chart frame
        self.chart_frame = ttk.Frame(result_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def set_sample_text(self, text):
        """Set sample text in the input field"""
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(tk.END, text)
    
    def analyze_text(self):
        """Analyze the input text"""
        text = self.text_input.get(1.0, tk.END).strip()

        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return

        if self.detector.model is None:
            messagebox.showwarning("Warning", "No model loaded. Please train or load a model first.")
            return

        try:
            self.status_var.set("Analyzing...")
            emotion, probs = self.detector.predict(text)

            # Update the emotion label
            self.emotion_label.config(text=emotion.upper())

            # Set the color for this emotion
            color = self.emotion_colors.get(emotion.lower(), self.default_color)
            self.emotion_label.config(foreground=color)

            # Create visualization
            self.create_emotion_chart(probs)

            self.status_var.set("Ready")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def create_emotion_chart(self, probs):
        """Create a simplified chart to visualize emotion probabilities"""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Sort probabilities and get top emotions
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_emotions = dict(sorted_probs[:5])  # Only show top 5

        # Create the figure
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)

        # Create horizontal bar plot for better readability
        emotions = list(top_emotions.keys())
        values = list(top_emotions.values())
        
        # Reverse the order for bottom-to-top display
        emotions.reverse()
        values.reverse()
        
        bars = ax.barh(emotions, values)

        # Color the bars based on emotions
        for i, emotion in enumerate(emotions):
            color = self.emotion_colors.get(emotion.lower(), self.default_color)
            bars[i].set_color(color)

        ax.set_title('Emotion Probabilities')
        ax.set_xlabel('Probability')
        plt.tight_layout()

        # Embed the plot
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store reference to avoid garbage collection
        self.fig = fig
        self.canvas = canvas