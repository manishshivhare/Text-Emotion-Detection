import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import neattext.functions as nfx

class TextPreprocessor:
    """Utility class for text preprocessing"""
    
    def __init__(self):
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = nfx.remove_urls(text)
        # Remove emails
        text = nfx.remove_emails(text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text and remove stopwords"""
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text):
        """Complete preprocessing pipeline"""
        clean_text = self.clean_text(text)
        tokens = self.tokenize(clean_text)
        lemmatized_tokens = self.lemmatize(tokens)
        
        return ' '.join(lemmatized_tokens)

# emotion_detector/ui/text_tab.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TextAnalysisTab:
    """Text analysis tab UI component"""
    
    def __init__(self, parent, detector, status_var):
        self.parent = parent
        self.detector = detector
        self.status_var = status_var
        self.fig = None
        self.canvas = None
        
        # Emotion colors
        self.emotion_colors = {
            'happy': '#FFD700',  # Gold
            'joy': '#FFD700',    # Gold
            'sad': '#4169E1',    # RoyalBlue
            'angry': '#FF4500',  # OrangeRed
            'anger': '#FF4500',  # OrangeRed
            'fear': '#800080',   # Purple
            'disgust': '#006400',  # DarkGreen
            'surprise': '#FF69B4',  # HotPink
            'neutral': '#A9A9A9',  # DarkGray
            'hate': '#8B0000',   # DarkRed
            'love': '#FF1493',   # DeepPink
        }
        self.default_color = '#1E90FF'  # DodgerBlue
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame for input
        input_frame = ttk.Frame(self.parent)
        input_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        # Label
        ttk.Label(input_frame, text="Enter text to analyze:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 5))

        # Text area for input
        self.text_input = tk.Text(input_frame, height=5, width=50, wrap=tk.WORD, font=('Segoe UI', 10))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)

        # Sample text options
        sample_frame = ttk.Frame(input_frame)
        sample_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sample_frame, text="Sample texts:").pack(side=tk.LEFT, padx=(0, 5))

        samples = [
            "I am so happy today!",
            "This makes me really angry!",
            "I'm feeling sad and down",
            "That's such a surprise!"
        ]

        for sample in samples:
            btn = ttk.Button(sample_frame, text=sample[:15] + "...",
                            command=lambda s=sample: self.set_sample_text(s))
            btn.pack(side=tk.LEFT, padx=2)

        # Analyze button
        ttk.Button(input_frame, text="Analyze Emotion", command=self.analyze_text).pack(pady=10)

        # Frame for results
        self.result_frame = ttk.Frame(self.parent)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Results heading
        ttk.Label(self.result_frame, text="Results:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 10))

        # Detected emotion
        emotion_frame = ttk.Frame(self.result_frame)
        emotion_frame.pack(fill=tk.X, pady=5)

        ttk.Label(emotion_frame, text="Primary emotion:").pack(side=tk.LEFT, padx=(0, 10))

        self.emotion_label = ttk.Label(emotion_frame, text="", style='Emotion.TLabel')
        self.emotion_label.pack(side=tk.LEFT)

        # Create a frame for the chart
        self.chart_frame = ttk.Frame(self.result_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
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
            self.status_var.set("Analyzing text...")
            emotion, probs = self.detector.predict(text)

            # Update the emotion label
            self.emotion_label.config(text=emotion.upper())

            # Get the color for this emotion
            color = self.emotion_colors.get(emotion.lower(), self.default_color)
            self.emotion_label.config(foreground=color)

            # Create visualization
            self.create_emotion_chart(probs)

            self.status_var.set("Analysis complete")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to analyze text: {str(e)}")
    
    def create_emotion_chart(self, probs):
        """Create a chart to visualize emotion probabilities"""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Sort probabilities in descending order
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_emotions = dict(sorted_probs[:5])  # Only show top 5

        # Create the figure
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)

        # Create bar plot
        bars = ax.bar(
            range(len(top_emotions)),
            list(top_emotions.values()),
            tick_label=list(top_emotions.keys())
        )

        # Color the bars based on emotions
        for i, (emotion, _) in enumerate(top_emotions.items()):
            color = self.emotion_colors.get(emotion.lower(), self.default_color)
            bars[i].set_color(color)

        ax.set_title('Emotion Probabilities')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Emotion')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store reference to avoid garbage collection
        self.fig = fig
        self.canvas = canvas