import pickle
import numpy as np
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print("Raw form data:", result)

        # Convert form data to dictionary
        res = result.to_dict(flat=True)
        print("Converted dict:", res)

        try:
            # Extract values and convert to float
            arr = [float(value) for value in res.values()]
            print("Converted array:", arr)

            # Convert to NumPy array
            data = np.array(arr, dtype=float).reshape(1, -1)
            print("Final numeric data:", data)
        except ValueError as e:
            return f"Error: Invalid input data. Ensure all inputs are numeric. {e}"

        # Load trained model
        try:
            with open("careerlast.pkl", 'rb') as model_file:
                loaded_model = pickle.load(model_file)
        except FileNotFoundError:
            return "Error: Model file 'careerlast.pkl' not found."
        except Exception as e:
            return f"Error loading model: {e}"

        # Make prediction
        try:
            predictions = loaded_model.predict(data)
            print("Prediction:", predictions)

            # Get probability predictions
            pred = loaded_model.predict_proba(data)
            print("Prediction probabilities:", pred)

            # Process predictions
            pred = pred > 0.05
            res = {}
            final_res = {}
            index = 0
            for j in range(17):
                if pred[0, j]:
                    res[index] = j
                    index += 1

            index = 0
            for key, values in res.items():
                if values != predictions[0]:
                    final_res[index] = values
                    index += 1

            # Job dictionary
            jobs_dict = {
                0: 'AI ML Specialist',
                1: 'API Integration Specialist',
                2: 'Application Support Engineer',
                3: 'Business Analyst',
                4: 'Customer Service Executive',
                5: 'Cyber Security Specialist',
                6: 'Data Scientist',
                7: 'Database Administrator',
                8: 'Graphics Designer',
                9: 'Hardware Engineer',
                10: 'Helpdesk Engineer',
                11: 'Information Security Specialist',
                12: 'Networking Engineer',
                13: 'Project Manager',
                14: 'Software Developer',
                15: 'Software Tester',
                16: 'Technical Writer'
            }

            # Get the predicted job
            predicted_job = jobs_dict.get(predictions[0], "Alternate śJob")
            

            return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=predicted_job)

        except Exception as e:
            return f"Error during prediction: {e}"

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
