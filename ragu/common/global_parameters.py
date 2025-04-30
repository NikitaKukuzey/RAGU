import logging
import os
from datetime import datetime

DEFAULT_FILENAMES = {
    "community_summary_kv_storage_name": "community_summary_kv.json",
    "chunks_kv_storage_name": "chunks_kv.json",
    "entity_vdb_name": "entity_vdb.json",
    "relation_vdb_name": "relation_vdb.json",
    "chunks_vdb_name": "chunks_vdb.json",
    "graph_name": "graph.gml",
}

current_dir = os.getcwd()
working_dir = os.path.join(current_dir, "ragu_working_dir")
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Folder for current run
current_run = os.path.join(working_dir, current_time)
os.makedirs(current_run, exist_ok=True)

# Folders that store knowledge graph outputs and indexer data
logs_dir = os.path.join(current_run, "logs")
run_output_dir = os.path.join(current_run, "outputs")
storage_run_dir = os.path.join(current_run, "storage")

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(run_output_dir, exist_ok=True)
os.makedirs(storage_run_dir, exist_ok=True)


