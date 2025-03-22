import os
from datetime import datetime

current_dir = os.getcwd()
working_dir = os.path.join(current_dir, "ragu_working_dir")
logs_dir = os.path.join(working_dir, "logs")
outputs_dir = os.path.join(working_dir, "outputs")

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
run_output_dir = os.path.join(outputs_dir, current_time)
os.makedirs(run_output_dir, exist_ok=True)