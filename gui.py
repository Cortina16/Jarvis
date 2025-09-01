import tkinter as tk
from tkinter import ttk, scrolledtext


class JarvisUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure the main window
        self.title("J.A.R.V.I.S.")
        self.geometry("1200x800")
        self.configure(bg="#0D1117")  # A dark theme color
        
        # TODO: Add the widgets and frames here
        # Use ttk.Style for a modern look
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#0D1117')
        self.style.configure('TLabel', background='#0D1117', foreground='#00A3FF', font=("Inter", 12))
        self.style.configure('TEntry', fieldbackground='#21262D', foreground='#FFFFFF', bordercolor='#30363D')
        self.style.configure('TButton', background='#161B22', foreground='#FFFFFF', font=("Inter", 12))
        
        # Add a main frame to hold all content with some padding
        self.main_content_frame = ttk.Frame(self)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Input frame at the bottom
        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)
        self.input_entry = ttk.Entry(self.input_frame, style='TEntry')
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 10))
        
        # Send button
        self.send_button = ttk.Button(self.input_frame, text="Send")
        self.send_button.pack(side=tk.LEFT, ipadx=10, ipady=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(self, bg="#161B22", fg="#FFFFFF", wrap=tk.WORD, height=10,
                                                  relief=tk.FLAT)
        self.log_text.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=20, pady=(0, 10))
        self.log_text.insert(tk.END, "J.A.R.V.I.S. is online. Awaiting command...\n")


if __name__ == "__main__":
    app = JarvisUI()
    app.mainloop()