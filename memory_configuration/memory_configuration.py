from flask import Flask, request, jsonify, render_template
from pathlib import Path
import json
import re

app = Flask(__name__)

# Get the directory of the currently running script
script_dir = Path(__file__).parent

# Construct the relative path to results.json
file_path = script_dir / '../../masters_thesis/python_code/results.json'

# Resolve the path to an absolute path (normalizing it)
file_path = file_path.resolve()

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

def convert_memory_value(value, num_ues, unit='kb'):
    if isinstance(value, str):
        match = re.match(r'([\d.]+)\s*(mb|kb)', value, re.IGNORECASE)
        if match:
            number, original_unit = match.groups()
            number = float(number)
            if original_unit.lower() == 'mb':
                number *= 1024  # Convert MB to KB
            # Multiply by number of UEs
            number *= num_ues
            # Convert back to MB if needed
            if unit.lower() == 'mb':
                number /= 1024
            return number
    return "no increase in memory after static memory"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the number of UEs and unit from the form
        num_ues = int(request.form.get('num_ues', 1))
        unit = request.form.get('unit', 'kb')

        # Deep copy of the original data
        modified_data = json.loads(json.dumps(data))

        # Modify the keys and multiply the memory values by num_ues
        for entity in modified_data:
            for item in modified_data[entity]:
                for key in list(modified_data[entity][item]):
                    value = modified_data[entity][item][key]
                    if key == 'memory per UE used':
                        new_key = f"memory for {num_ues} UEs needed ({unit})"
                        modified_data[entity][item][new_key] = convert_memory_value(value, num_ues, unit)
                        del modified_data[entity][item][key]
                    elif key == 'buffer memory':
                        new_key = f"buffer memory for {num_ues} UEs needed ({unit})"
                        modified_data[entity][item][new_key] = convert_memory_value(value, num_ues, unit)
                        del modified_data[entity][item][key]

        return jsonify(modified_data)
    else:
        # Render the HTML form from the template
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
