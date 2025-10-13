import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mplfinance as mpf
import numpy as np
import matplotlib.dates as mdates


class CandlestickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Candlestick Chart")
        self.root.geometry("1000x800")

        # Frame for plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X)

        load_button = tk.Button(button_frame, text="Load CSV", command=self.load_csv)
        load_button.pack(side=tk.LEFT, padx=10, pady=10)

        quit_button = tk.Button(button_frame, text="Quit", command=self.root.quit)
        quit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.df = None
        self.fig = None
        self.canvas = None
        self.toolbar = None
        self.annot = None
        self.connection = None

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            self.df = self.df.sort_index()
            self.plot_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def plot_chart(self):
        # Clear existing plot if any
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()
        if self.connection:
            self.fig.canvas.mpl_disconnect(self.connection)
        if self.annot:
            self.annot.remove()

        # Create figure with subplots if Volume present
        has_volume = 'Volume' in self.df.columns
        fig, axlist = mpf.plot(
            self.df,
            type='candle',
            style='charles',
            volume=has_volume,
            returnfig=True,
            figsize=(10, 7) if has_volume else (10, 5),
            title='Interactive Candlestick Chart'
        )

        self.fig = fig
        self.ax = axlist[0]  # Main axis for candlesticks

        # Precompute date numbers for efficient lookup
        self.dates_num = mdates.date2num(self.df.index)

        # Create annotation for hover tooltip
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", ec="k", lw=1),
            arrowprops=dict(arrowstyle="->")
        )
        self.annot.set_visible(False)

        # Connect motion event
        self.connection = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def on_motion(self, event):
        if event.inaxes != self.ax:
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        x = event.xdata
        # Find closest index
        idx = np.argmin(np.abs(self.dates_num - x))
        data = self.df.iloc[idx]

        # Format text
        text = f"Date: {data.name.strftime('%Y-%m-%d')}\n"
        text += f"Open: {data['Open']:.2f}\n"
        text += f"High: {data['High']:.2f}\n"
        text += f"Low: {data['Low']:.2f}\n"
        text += f"Close: {data['Close']:.2f}"
        if 'Volume' in data:
            text += f"\nVolume: {data['Volume']:.0f}"

        # Update annotation
        self.annot.xy = (x, data['High'])
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.8)
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = CandlestickApp(root)
    root.mainloop()