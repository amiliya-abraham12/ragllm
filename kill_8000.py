import psutil
for conn in psutil.net_connections():
    if conn.laddr.port == 8000:
        print(f"Killing process {conn.pid}")
        try:
            p = psutil.Process(conn.pid)
            p.terminate()
            p.wait(timeout=5)
            print("Terminated")
        except Exception as e:
            print("Failed to terminate:", e)
