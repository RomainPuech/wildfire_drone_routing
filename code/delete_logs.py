import os
import shutil

def clear_logs_in_dronebench(base_dir="../DroneBench"):
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    layout_folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for layout in layout_folders:
        logs_dir = os.path.join(layout, "logs")
        if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
            print(f"Clearing logs in: {logs_dir}")
            for filename in os.listdir(logs_dir):
                file_path = os.path.join(logs_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"No logs directory in {layout}")

if __name__ == "__main__":
    clear_logs_in_dronebench()