from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

data = pd.read_csv(r"C:\Users\MY PC\Downloads\new_datasetcsv (3).csv")
df = pd.DataFrame(data)
df.columns = df.columns.str.strip()

categorical_cols = ['City', 'Type']
numeric_cols = ['Year', 'Population']
target_col = 'Crime Rate'

overall_mean = df[target_col].mean()

# ---- Step 1: per-city linear trend (this part CAN extrapolate to future years) ----
trend_models = {}
for city in df['City'].unique():
    sub = df[df['City'] == city]
    if len(sub) >= 2:  # need at least 2 points to fit a line
        lr = LinearRegression()
        lr.fit(sub[['Year']], sub[target_col])
        trend_models[city] = lr
    else:
        trend_models[city] = None  # not enough data, will fall back to overall mean

def trend_component(city, year):
    model = trend_models.get(city)
    if model is not None:
        return model.predict([[year]])[0]
    return overall_mean

# ---- Step 2: residuals = actual rate minus what the trend alone predicts ----
df['trend_pred'] = df.apply(lambda r: trend_component(r['City'], r['Year']), axis=1)
df['residual'] = df[target_col] - df['trend_pred']

# ---- Step 3: Random Forest learns the residual (City/Type/Population effects) ----
X = df[categorical_cols + numeric_cols]
y_resid = df['residual']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

residual_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
residual_model.fit(X, y_resid)

# ---- Step 4: final prediction = trend + residual, floored at 0 ----
def predict_crime_rate(Year, City, Population, Type):
    Year = float(Year)
    Population = float(Population)

    trend_part = trend_component(City, Year)

    new_df = pd.DataFrame({
        'City': [City], 'Type': [Type],
        'Year': [Year], 'Population': [Population]
    })
    residual_part = residual_model.predict(new_df)[0]

    prediction = trend_part + residual_part
    return max(0, round(prediction, 2))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction = predict_crime_rate(
            data['Year'], data['City'], data['Population'], data['Type']
        )
        return jsonify({'predicted_rate': prediction})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


# ---- NEW: powers the "Trend Analysis" chart on the frontend ----
@app.route('/history', methods=['POST'])
def history():
    """Returns historical Crime Rate by Year for a given City (optionally
    filtered by Type) as a plain JSON array: [{"year": ..., "rate": ...}, ...]
    This is exactly what the frontend's fetchHistory() function expects."""
    try:
        data = request.get_json()
        city = data.get('City')
        crime_type = data.get('Type')

        if not city:
            return jsonify({'error': 'City is required'}), 400

        sub = df[df['City'] == city]
        if crime_type:
            sub = sub[sub['Type'] == crime_type]

        if sub.empty:
            return jsonify([])  # frontend handles an empty array gracefully

        grouped = sub.groupby('Year')[target_col].mean().reset_index()
        grouped = grouped.sort_values('Year')

        series = [
            {'year': int(row['Year']), 'rate': round(float(row[target_col]), 2)}
            for _, row in grouped.iterrows()
        ]
        return jsonify(series)

    except Exception as e:
        print("History error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)