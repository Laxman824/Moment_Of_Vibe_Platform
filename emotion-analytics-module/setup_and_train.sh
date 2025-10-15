#!/bin/bash
# setup_and_train.sh

echo "=== Setting up Emotion Analytics Module ==="

# 1. Organize data
echo "1. Organizing RAVDESS data..."
if [ -d "Audio_Speech_Actors_01-24" ]; then
    echo "   Moving RAVDESS files to data/raw..."
    mkdir -p data/raw
    cp -r Audio_Speech_Actors_01-24/* data/raw/ 2>/dev/null || true
    echo "   ✓ Data organized"
else
    echo "   ⚠ RAVDESS folder not found"
fi

# 2. Prepare dataset
echo -e "\n2. Preparing dataset..."
python3 scripts/prepare_data.py

# 3. Train model
echo -e "\n3. Training model..."
python3 scripts/train.py

# 4. Run demo again
echo -e "\n4. Running demo with trained model..."
./demo_script.sh