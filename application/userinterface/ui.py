import sys
import os
import shutil
import logging
from pathlib import Path

from ultralytics import YOLO

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QWidget, QFileDialog, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from multiprocessing import Queue
import time

from forest_elephants_rumble_detection.model.yolo.predict import pipeline
from forest_elephants_rumble_detection.utils import yaml_read, yaml_write


# Worker class to handle processing in a separate thread
class Worker(QThread):
    # Signal to update progress (file index, progress percentage)
    progress = pyqtSignal(int, int)  
    finished = pyqtSignal(int, str)  # Signal to indicate processing finished (file index, output path)

    def __init__(self, file_index, file_paths, input_dir, output_dir, queue):
        super().__init__()
        self.file_index = file_index  # Index of the file being processed
        self.file_paths = file_paths    # Path of the file being processed
        self.output_dir = output_dir
        self.input_dir = input_dir # Directory to save processed files
        self.queue = queue            # Queue to communicate progress

    def run(self):
            
        config = yaml_read(Path("./application/config.yaml"))

        logging.basicConfig(level=config["loglevel"].upper())

        model = YOLO(config["model_weights_filepath"])

        for i, file_path in enumerate(self.file_paths):
            df_pipeline = pipeline(
                model=model,
                audio_filepaths=[file_path],
                duration=config["duration"],
                overlap=config["overlap"],
                width=config["width"],
                height=config["height"],
                freq_min=config["freq_min"],
                freq_max=config["freq_max"],
                n_fft=config["n_fft"],
                hop_length=config["hop_length"],
                batch_size=config["batch_size"],
                output_dir=self.output_dir,
                save_spectrograms=config["save_spectrograms"],
                save_predictions=config["save_predictions"],
                verbose=config["verbose"],
            )

            logging.info(f"Saving the results")
            logging.info(df_pipeline.head())
            df_pipeline.to_csv(self.output_dir / "results.csv")
            yaml_write(
                self.output_dir / "args.yaml",
                {
                    "config": {**config},
                    "args": {
                        "batch_size": config["batch_size"],
                        "overlap": config["overlap"],
                        "model_weights_filepath": str(config["model_weights_filepath"]),
                        "input_dir_audio_filepaths": self.input_dir,
                    },
                },
            )

        self.finished.emit(self.file_index, str(self.output_dir))  # Emit finished signal with output path


# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.input_dir = Path()
        self.output_dir = Path()  # Initialize output_dir attribute

    def initUI(self):
        self.setWindowTitle('Audio Data Processor')  # Set window title
        self.setGeometry(100, 100, 400, 300)         # Set window size and position

        layout = QVBoxLayout()  # Create a vertical box layout

        # Label to display the selected file information
        self.file_label = QLabel("No files selected")
        layout.addWidget(self.file_label)

        # Button to select input directory
        self.input_button = QPushButton("Select input Directory")
        self.input_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.input_button)
        
        # Button to select output directory
        self.output_button = QPushButton("Select Output Directory")
        self.output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.output_button)

        # Label to display the selected output directory
        self.input_label = QLabel("No input directory selected")
        layout.addWidget(self.input_label)

        # Label to display the selected output directory
        self.output_label = QLabel("No output directory selected")
        layout.addWidget(self.output_label)

        # Button to start processing files, initially disabled
        self.process_button = QPushButton("Process Files")
        self.process_button.setEnabled(False)
        
        self.process_button.clicked.connect(self.process_files)
        layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar(self)  # List to hold progress bars for each file
        layout.addWidget(self.progress_bar)

        self.time_label = QLabel("Processing Time: 0 seconds")
        layout.addWidget(self.time_label)


        # Container widget to hold the layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.selected_files = []  # List to hold selected file paths
    
    def select_output_directory(self):
        # Open a directory dialog to select the output directory
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.output_label.setText(f"Output directory: {directory}")
            self.update_process_button_state()  # Checking that files and output directory are specified

    def parse_wav_files(self, folder):
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.endswith(".wav"):
                    file_path = Path(root) / filename 
                    self.selected_files.append(file_path)

    def select_input_directory(self):
        # Open a directory dialog to select the output directory
        input_directory = QFileDialog.getExistingDirectory(self, "Select input Directory")
        if input_directory:
            self.input_dir = Path(input_directory)
            self.input_label.setText(f"Input directory: {input_directory}")
            self.update_process_button_state()  # Checking that files and output directory are specified
            self.parse_wav_files(input_directory)
            self.file_label.setText(f"Selected {len(self.selected_files)} files")

    def update_process_button_state(self):
        # Enable the process button only if files are selected and output directory is set
        if (self.selected_files or self.input_dir) and self.output_dir:
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)

    def process_files(self):
        self.start_time = time.time()  # Record start time
        self.queue = Queue()
        self.workers = []

        batch_size = 1
        file_batches = [self.selected_files[i:i + batch_size] for i in range(0, len(self.selected_files), batch_size)]

        self.total_files = len(self.selected_files)
        self.completed_files = 0
        self.progress_bar.setValue(0)

        for index, file_batch in enumerate(file_batches):
            worker = Worker(index, file_batch, self.input_dir, self.output_dir, self.queue)
            worker.finished.connect(self.file_finished)
            self.workers.append(worker)
            worker.start()



    def file_finished(self, file_index, output_path):
        self.completed_files += 1
        progress = int((self.completed_files / self.total_files) * 100)
        self.progress_bar.setValue(progress)
        self.output_label.setText(f"Processed {self.completed_files}/{self.total_files} files")

        if self.completed_files == self.total_files:
            elapsed_time = time.time() - self.start_time  # Calculate elapsed time
            self.time_label.setText(f"Processing Time: {int(elapsed_time)} seconds")  # Update the label
            self.process_button.setEnabled(True)
            self.output_label.setText(f"All files processed. Output saved to: {output_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create a QApplication
    mainWindow = MainWindow()     # Create an instance of MainWindow
    mainWindow.show()             # Show the main window
    sys.exit(app.exec_())         # Run the application's event loop
