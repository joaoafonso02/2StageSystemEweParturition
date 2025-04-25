import joblib
import emlearn
import numpy as np
import os

# Load the model
model_path = 'best_aggregate_1Hz_75min.pkl'
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Convert to emlearn format
print("Converting model to emlearn format...")
eml_model = emlearn.convert(model)

# Define valid C identifier and output file path
c_identifier_name = 'model_eml'
output_filename = 'model_eml.h'

print(f"Saving model to file: {output_filename} with C identifier: {c_identifier_name}")
with open(output_filename, 'w') as f:
    f.write(eml_model.save(name=c_identifier_name))

# Confirm save
if os.path.exists(output_filename):
    print(f"Model successfully saved to: {output_filename}")
else:
    print("Failed to save model.")

