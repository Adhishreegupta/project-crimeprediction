from flask import Flask, request, jsonify 
from flask_cors import CORS 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 

app = Flask(__name__)
CORS(app)
# Load and prepare the data
data = pd.read_csv(r"C:\Users\MY PC\Downloads\new_datasetcsv (3).csv")
df = pd.DataFrame(data)


df.columns = df.columns.str.strip()
print(df.columns.tolist())

categorical_cols = ['City', 'Type']
numeric_cols = ['Year', 'Population']
target_col = 'Crime Rate'
X = df[categorical_cols + numeric_cols]
y = df[target_col]
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X, y)

def predict_crime_rate(Year,City,Population,Type):
    """Predicts the crime rate using the trained model."""
    new_data = {
        'City': [City],
        'Type': [Type],
        'Year': [float(Year)],
        'Population': [float(Population)]
    }
    new_df = pd.DataFrame(new_data)
    prediction = model.predict(new_df)
    return prediction[0]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Input data:", data)

        Year = float(data['Year'])
        City = data['City']
        Population = float(data['Population'])
        Type = data['Type']

        new_df = pd.DataFrame({
            'City': [City],
            'Type': [Type],
            'Year': [Year],
            'Population': [Population]
        })

        prediction = model.predict(new_df)
        return jsonify({'predicted_rate': round(prediction[0], 2)})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)