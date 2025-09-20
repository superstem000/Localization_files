# Calculate the shared start time (current Unix timestamp)
START_TIME=$(python -c "import time; print(time.time())")

# Run two instances with the same start_time in PowerShell windows
start powershell -Command "cd 'C:\Users\bhpar\Desktop\CS\Sound-Project\respeaker_curr'; bash -i record.sh 2 3 $START_TIME; bash -i record.sh 3 2 $START_TIME; pause"


