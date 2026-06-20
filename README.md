# Crime Rate Predictor

A machine learning project that predicts crime rates for a given city, year, population, and offense type — paired with a custom-built analytics dashboard for visualizing predictions against historical trends.

## Overview

This project started as a simple Linear Regression model and evolved through several iterations to address real limitations found along the way:

1. **Random Forest Regressor** — fixed the negative-prediction issue (outputs are bounded by training data), but couldn't extrapolate to future years.
2. **Hybrid Trend + Random Forest model** — combines a per-city linear trend (captures the direction crime is moving over time) with a Random Forest residual model (captures how City, Type, and Population interact non-linearly). This is the final model used in production.

## Features

- Predicts crime rate from **City**, **Year**, **Population**, and **Crime Type**
- Forecasts future years using a trend-aware hybrid model (not just a flat repeated value)
- REST API backend built with Flask
- Custom-themed frontend dashboard — a "Crime Analytics Dispatch Terminal" — featuring:
  - A live risk gauge (LOW / ELEVATED / HIGH)
  - An animated printed case-slip showing the prediction result
  - A historical trend chart comparing the forecast against past years for the selected city/type

## Tech Stack

**Backend:** Python, Flask, Flask-CORS, pandas, scikit-learn
**Frontend:** HTML, CSS, JavaScript, [Chart.js](https://www.chartjs.org/)

