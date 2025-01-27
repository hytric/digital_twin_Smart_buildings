# Electricity Usage Prediction with LSTM

This project predicts electricity usage for the next 6 hours using an LSTM model. It processes data from **electricity.csv** and **weather.csv** (air temperature) to forecast usage for multiple buildings in the **Building Data Genome Project 2** dataset.

## Key Features
- **Dataset**: Electricity usage and weather data filtered for 2016-2017.
- **Model**: LSTM architecture for time-series prediction.
- **Forecast Horizon**: Predicts the next 6 hours of electricity usage.
- **Validation**:
  - ACF/ADF tests performed on the time series.
  - Preprocessing includes outlier removal and interpolation.
- **Visualization**: UI displays predictions vs. actuals with temperature trends.
- **Results**:
  - Example: `Panther_education_Teofila` - MAE=17.768, RMSE=24.849, MAPE=11.34%.
  - Warnings indicate prediction threshold violations.

## Usage

### 1. Data Preparation
Ensure the `electricity.csv` and `weather.csv` files are in the `./data/archive/` directory. Preprocess the data to remove outliers and fill missing values.

### 2. Model Training
Train the LSTM model for each building:

```bash
python education_electric.py
```

Results and trained models are saved in the ./models directory.

3. Prediction

Use the app.py to make predictions via the Streamlit UI:

```bash
streamlit run app.py
```


4. Visualization

View actual vs. predicted usage, with a focus on:
	•	18 hours of historical data.
	•	6 hours of future predictions.


---


Example Results

Below are the detailed results for each building:
## Example Results

| Building                   | MAE     | RMSE    | MAPE (%)        | Warnings |
|----------------------------|---------|---------|-----------------|----------|
| Panther_education_Teofila  | 17.768  | 24.849  | 11.34           | 309      |
| Panther_education_Misty    | 2.525   | 3.938   | 8.88            | 589      |
| Panther_education_Tina     | 1.370   | 1.909   | 24.58           | 518      |
| Panther_education_Janis    | 4.029   | 5.897   | 15.60           | 685      |
| Panther_education_Quintin  | 43.612  | 58.142  | 15.38           | 407      |
| Panther_education_Violet   | 6.064   | 8.197   | 4.30            | 167      |
| Panther_education_Edna     | 13.791  | 18.489  | 12.51           | 504      |
| Panther_education_Sophia   | 7.473   | 8.954   | 4.84            | 41       |
| Panther_education_Annetta  | 14.790  | 18.951  | 14.62           | 444      |
| Panther_education_Ivan     | 15.311  | 19.869  | 11.69           | 421      |
| Panther_education_Alecia   | 32.914  | 41.768  | 19.35           | 662      |
| Panther_education_Rosalie  | 4.869   | 6.551   | 7.94            | 485      |
| Panther_education_Jonathan | 0.840   | 1.149   | 28486562.50     | 828      |
---

<br>

## Dependencies  
•	Python (3.8+)  
•	PyTorch  
•	pandas, numpy, matplotlib, seaborn  
•	Streamlit for visualization

## Future Improvements
•	Expand feature engineering for better accuracy.  
•	Support for additional buildings and environmental factors.

## References
•	Dataset: The Building Data Genome Project 2  

Feel free to contribute by opening an issue or submitting a pull request.