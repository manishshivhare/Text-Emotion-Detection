import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from sklearn.model_selection import train_test_split

from ..models.emotion_detector import EmotionDetector
from .text_tab import TextAnalysisTab


class EmotionDetectorUI:
    """Simplified main application UI class"""

    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("700x500")  # Reduced size for simpler UI

        # Initialize detector
        self.detector = EmotionDetector()

        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create UI components
        self.create_menu()
        self.create_main_frame()

        # Try to load model
        self.load_model()

    def create_menu(self):
        """Create simplified application menu"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self.load_model_from_dialog)
        file_menu.add_command(label="Train Model", command=self.show_train_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_main_frame(self):
        """Create main content frame"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup text analysis in the main frame
        self.text_analysis = TextAnalysisTab(main_frame, self.detector, self.status_var)

    def load_model(self):
        """Load the default model"""
        try:
            self.detector.load_model()
            self.status_var.set("Model loaded")
        except FileNotFoundError:
            self.status_var.set("No model found. Please train or load a model.")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")

    def load_model_from_dialog(self):
        """Load a model from a directory selected via dialog"""
        model_dir = filedialog.askdirectory(
            title="Select Model Directory",
            initialdir=os.getcwd()
        )

        if model_dir:
            try:
                self.detector.load_model(model_dir)
                self.status_var.set(f"Model loaded from {model_dir}")
            except Exception as e:
                self.status_var.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def show_about(self):
        """Display about dialog"""
        messagebox.showinfo(
            "About",
            "Emotion Detection System\n\n"
            "This application detects emotions in text using machine learning."
        )

    def show_train_dialog(self):
        """Show simplified dialog for training a new model"""
        # Create training dialog
        train_dialog = tk.Toplevel(self.root)
        train_dialog.title("Train Model")
        train_dialog.geometry("350x200")  # Smaller dialog
        train_dialog.transient(self.root)
        train_dialog.grab_set()
        
        # Dataset selection
        ttk.Label(train_dialog, text="Select Dataset:").pack(anchor=tk.W, padx=10, pady=(10, 5))

        file_frame = ttk.Frame(train_dialog)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        dataset_path_var = tk.StringVar()
        dataset_entry = ttk.Entry(file_frame, textvariable=dataset_path_var, width=30)
        dataset_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        def browse_file():
            file_path = filedialog.askopenfilename(
                title="Select Dataset",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                dataset_path_var.set(file_path)
        
        ttk.Button(file_frame, text="Browse", command=browse_file).pack(side=tk.LEFT, padx=5)

        # Save path
        save_frame = ttk.Frame(train_dialog)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(save_frame, text="Save as:").pack(side=tk.LEFT, padx=(0, 5))
        
        save_path_var = tk.StringVar(value="emotion_model")
        ttk.Entry(save_frame, textvariable=save_path_var, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Progress indicator
        progress_var = tk.StringVar(value="Ready to train")
        progress_label = ttk.Label(train_dialog, textvariable=progress_var)
        progress_label.pack(fill=tk.X, padx=10, pady=10)

        # Function to train model
        def train_model_thread():
            """Separate thread for model training"""
            # Disable buttons during training
            train_btn.config(state=tk.DISABLED)
            cancel_btn.config(state=tk.DISABLED)
            progress_var.set("Training...")

            try:
                dataset_path = dataset_path_var.get()
                
                if not dataset_path:
                    progress_var.set("Error: No dataset selected")
                    messagebox.showerror("Error", "Please select a dataset file.")
                    train_btn.config(state=tk.NORMAL)
                    cancel_btn.config(state=tk.NORMAL)
                    return
                
                # Load and process data
                progress_var.set("Loading dataset...")
                df = self.detector.load_data(dataset_path)
                
                # Split data
                progress_var.set("Preparing training data...")
                X_train, X_test, y_train, y_test = train_test_split(
                    df['clean_text'], df['emotion'], test_size=0.2, random_state=42
                )
                
                # Train model
                progress_var.set("Training model...")
                self.detector.train(X_train, y_train, model_type="nb")
                
                # Evaluate
                progress_var.set("Evaluating model...")
                accuracy, _, _ = self.detector.evaluate(X_test, y_test)
                progress_var.set(f"Model accuracy: {accuracy:.2f}")
                
                # Save model
                save_path = save_path_var.get()
                progress_var.set(f"Saving model...")
                self.detector.save_model(save_path)
                
                progress_var.set("Training complete!")
                self.status_var.set(f"Model trained and saved")
                
                # Close dialog after a delay
                train_dialog.after(1500, lambda: self.close_train_dialog(train_dialog))

            except Exception as error:
                progress_var.set(f"Error: {str(error)}")
                messagebox.showerror("Error", f"Training failed: {str(error)}")
                
                # Enable buttons
                train_btn.config(state=tk.NORMAL)
                cancel_btn.config(state=tk.NORMAL)

        # Function to start training in a separate thread
        def start_training():
            training_thread = threading.Thread(target=train_model_thread)
            training_thread.daemon = True
            training_thread.start()

        # Buttons
        btn_frame = ttk.Frame(train_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        train_btn = ttk.Button(btn_frame, text="Train", command=start_training)
        train_btn.pack(side=tk.RIGHT, padx=5)

        cancel_btn = ttk.Button(btn_frame, text="Cancel", 
                               command=lambda: self.close_train_dialog(train_dialog))
        cancel_btn.pack(side=tk.RIGHT, padx=5)
    
    def close_train_dialog(self, dialog):
        """Close the training dialog"""
        dialog.grab_release()
        dialog.destroy()