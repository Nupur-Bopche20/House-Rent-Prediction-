# House Rent Prediction Web App

This is a full-stack machine learning project that predicts house rent prices in major Indian cities based on user-provided features.

The project consists of two parts:
1.  A Python machine learning model (using LightGBM) trained on a dataset of over 4,700 house listings.
2.  A web application (HTML/CSS/JS) with a Flask server back-end that serves the trained model via a REST API.

**Final Model Accuracy (R-squared): 85.44%**

---

### Tech Stack

* **Back-End (Model & API):**
    * Python
    * Flask (for the REST API)
    * scikit-learn (for preprocessing pipelines and metrics)
    * LightGBM (for the regression model)
    * Pandas (for data cleaning and feature engineering)
    * Joblib (for saving/loading the trained model)

* **Front-End (Website):**
    * HTML
    * Tailwind CSS
    * JavaScript (for dynamic dropdowns and API calls)

---

### Features

* **Accurate ML Model:** Uses a LightGBM regressor, which achieved an 85.44% R-squared score on the test data.
* **Intelligent Feature Engineering:**
    * Parses complex strings (like "Ground out of 4") into usable numerical features (`Floor_Num`, `Total_Floors`).
    * Groups over 2,200 unique "Area Localities" into a "Top 50 + Other" feature to improve accuracy without overfitting.
* **Dynamic Front-End:** The "Area Locality" dropdown automatically updates based on the "City" selected by the user.
* **Full-Stack Workflow:** The front-end sends user data to the Flask API, which processes the data, feeds it to the model, and returns a real-time prediction.

---

### How to Run This Project Locally

**1. Clone the Repository:**
```bash
git clone [https://github.com/Nupur-Bopche20/House-Rent-Prediction-.git](https://github.com/Nupur-Bopche20/House-Rent-Prediction-.git)
cd House-Rent-Prediction-
