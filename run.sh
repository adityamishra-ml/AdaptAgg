#!/bin/bash

echo "🚀 Starting federated learning simulation for YOLOv8..."
# Number of clients to simulate
NUM_CLIENTS=6

# Kill any existing processes to ensure a clean start
echo "🧹 Cleaning up old processes..."
pkill -f server.py
pkill -f client.py

# 1. Start the server in the background
echo "🖥️  Starting server..."
python3 server.py &
SERVER_PID=$!
sleep 5 # Give the server a moment to start

# 2. Start all clients in the background
echo "👥 Starting $NUM_CLIENTS clients..."
for i in $(seq 0 $(($NUM_CLIENTS - 1)))
do
    echo "   - Starting client $i"
    python3 client.py --cid $i &
done

# Wait for all background processes to complete
echo "⏳ All processes launched. Waiting for training to finish..."
wait $SERVER_PID
echo "✅ Federated learning simulation complete."
