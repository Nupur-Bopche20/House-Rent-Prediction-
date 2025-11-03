import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
print("Loading dataset...")
try:
    df = pd.read_csv('House_Rent_Dataset.csv')
except FileNotFoundError:
    print("Error: House_Rent_Dataset.csv not found.")
    exit()

# --- 2. Cleaning (from our notebook) ---
df.drop_duplicates(inplace=True)
df = df[df['Size'] >= 100]

# --- 3. Feature Engineering (from our notebook) ---

# Function to process the 'Floor' string
def process_floor(floor_str):
    if pd.isna(floor_str):
        return np.nan, np.nan
    
    parts = floor_str.split(' out of ')
    floor_num = np.nan
    total_floors = np.nan
    
    if len(parts) == 2:
        try:
            total_floors = int(parts[1])
        except ValueError:
            total_floors = np.nan 
    
    floor_val = parts[0].lower().strip()
    if 'ground' in floor_val:
        floor_num = 0
    elif 'upper basement' in floor_val:
        floor_num = -1
    elif 'lower basement' in floor_val:
        floor_num = -2
    else:
        try:
            floor_num = int(floor_val)
        except ValueError:
            floor_num = np.nan
            
    return floor_num, total_floors

# Apply floor processing
df[['Floor_Num', 'Total_Floors']] = df['Floor'].apply(lambda x: pd.Series(process_floor(x)))

# Fill missing floor values
median_total_floors = df['Total_Floors'].median()
df['Total_Floors'] = df['Total_Floors'].fillna(median_total_floors)
median_floor_num = df['Floor_Num'].median()
df['Floor_Num'] = df['Floor_Num'].fillna(median_floor_num)

# Create engineered floor features
df['Is_Top_Floor'] = (df['Floor_Num'] == df['Total_Floors']).astype(int)
df['Is_Ground_Floor'] = (df['Floor_Num'] == 0).astype(int)

# Create 'Area_Locality_Simple' (our best feature set)
top_50_localities = df['Area Locality'].value_counts().head(50).index.tolist()
df['Area_Locality_Simple'] = df['Area Locality'].apply(lambda x: x if x in top_50_localities else 'Other')

# Create 'Rent_Log' (our target)
df['Rent_Log'] = np.log1p(df['Rent'])

print("Feature engineering complete.")

# --- 4. Define Final Features and Preprocessor ---

numerical_features = [
    'BHK', 'Size', 'Bathroom', 'Floor_Num', 'Total_Floors', 
    'Is_Top_Floor', 'Is_Ground_Floor'
]

categorical_features_v3 = [
    'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 
    'Area_Locality_Simple'
]

# Define our features (X) and target (y)
all_features = numerical_features + categorical_features_v3
X = df[all_features]
y = df['Rent_Log']

# Create the V3 Preprocessor
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer_v3 = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor_v3 = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer_v3, categorical_features_v3)
    ])

print("Preprocessor 'preprocessor_v3' created.")

# --- 5. Define and Train the Final Model ---
# This is our best model pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_v3),
    ('regressor', lgb.LGBMRegressor(random_state=42, n_jobs=-1))
])

print("Training the final model on the *entire* dataset...")
# We train on ALL data for deployment
final_pipeline.fit(X, y)
print("Model training complete.")

# --- 6. Save the Model ---
model_filename = 'house_rent_predictor.joblib'
joblib.dump(final_pipeline, model_filename)

print(f"\n--- SUCCESS! ---")
print(f"Your best model has been trained and saved as '{model_filename}'")
print("\nYou would now load this file into a Python server (like Flask) to make predictions.")

# Also print the list of localities for our HTML file
print("\n--- Top 50 Localities (for website dropdown) ---")
print(['Other'] + top_50_localities)
