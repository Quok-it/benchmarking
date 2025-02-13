import pandas as pd

# Load the Excel file with multi-index headers
file_path = "ai-benchmark-results.xlsx"  # Update with actual file path
df = pd.read_excel(file_path, header=[0, 1])  # Read first two rows as header

# Display the first few rows
print(df.head())
# Flatten MultiIndex columns by joining them with a space
df.columns = [' '.join(col).strip() for col in df.columns]

# Print cleaned column names
print(df.columns)

# Initialize dictionary
ai_benchmark_results = {}

# Extract GPU model names (assuming first column is GPU names)
gpu_column = df.columns[0]

# Iterate through the cleaned columns
for col in range(1, len(df.columns), 2):  # Step by 2 to handle inference & training
    network_name = df.columns[col].rsplit(' ', 1)[0]  # Extract network name
    network_name = network_name.replace(" Inference", "")
    inference_time_col = df.columns[col]  # Column for inference time
    training_time_col = df.columns[col + 1]  # Column for training time

    # Ensure network entry exists in the dictionary
    if network_name not in ai_benchmark_results:
        ai_benchmark_results[network_name] = {"Inference time": {}, "Training time": {}}

    # Iterate over GPU models
    for _, row in df.iterrows():
        gpu_model = row[gpu_column]  # GPU model
        inference_time = row[inference_time_col]  # Inference time
        training_time = row[training_time_col]  # Training time

        # Store data
        ai_benchmark_results[network_name]["Inference time"][gpu_model] = inference_time
        ai_benchmark_results[network_name]["Training time"][gpu_model] = training_time

# Print the structured dictionary
import json
print(json.dumps(ai_benchmark_results, indent=4))
