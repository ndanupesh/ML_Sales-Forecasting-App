import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime as dt


# Cache model loading for speed
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('sales_forecast_rf_model_tuned.pkl')
    return model


rf_model = load_model()


st.title("Advanced Sales Forecasting App")


# Sidebar controls
st.sidebar.header("Settings and Controls")
show_actuals = st.sidebar.checkbox("Show Actual Sales in Plots", value=True)
default_stock = st.sidebar.number_input("Default Stock for future predictions", min_value=0, value=100)
default_price = st.sidebar.number_input("Default Price for future predictions", min_value=0.0, value=20.0, format="%.2f")


uploaded_file = st.file_uploader("Upload your sales data CSV file", type=['csv'])


def preprocess_data(data):
    data = data.copy()
    data.rename(columns={
        'data': 'Date',
        'venda': 'Sales',
        'estoque': 'Stock',
        'preco': 'Price'
    }, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['Sales_Lag7'] = data['Sales'].shift(7)
    data['Sales_Lag30'] = data['Sales'].shift(30)
    data['Sales_7d_Mean'] = data['Sales'].rolling(7).mean()
    data['Sales_30d_Mean'] = data['Sales'].rolling(30).mean()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def build_features_for_date(date, processed_df, default_stock, default_price):
    features_dict = {
        'Month': date.month,
        'DayOfWeek': date.weekday(),
        'IsWeekend': int(date.weekday() >= 5)
    }
    feature_row = processed_df[processed_df['Date'] == pd.to_datetime(date)]
    if not feature_row.empty:
        features_dict['Sales_Lag7'] = feature_row['Sales_Lag7'].values[0]
        features_dict['Sales_Lag30'] = feature_row['Sales_Lag30'].values[0]
        features_dict['Sales_7d_Mean'] = feature_row['Sales_7d_Mean'].values[0]
        features_dict['Sales_30d_Mean'] = feature_row['Sales_30d_Mean'].values[0]
        features_dict['Stock'] = feature_row['Stock'].values[0]
        features_dict['Price'] = feature_row['Price'].values[0]
    else:
        features_dict['Sales_Lag7'] = 0
        features_dict['Sales_Lag30'] = 0
        features_dict['Sales_7d_Mean'] = 0
        features_dict['Sales_30d_Mean'] = 0
        features_dict['Stock'] = default_stock
        features_dict['Price'] = default_price
    return pd.DataFrame([features_dict])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    processed_df = preprocess_data(df)
    st.subheader("Preprocessed data sample")
    st.dataframe(processed_df.head())

    features = ['Month', 'DayOfWeek', 'IsWeekend',
                'Sales_Lag7', 'Sales_Lag30', 'Sales_7d_Mean', 'Sales_30d_Mean',
                'Stock', 'Price']

    # Check for missing features
    missing = [f for f in features if f not in processed_df.columns]
    if missing:
        st.error(f"Missing features after preprocessing: {missing}")
        st.stop()

    # Predict on historical data
    X = processed_df[features]
    predictions = rf_model.predict(X)
    processed_df['Forecasted_Sales'] = predictions

    # Create tabs for organizing UI
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview & Forecast",
        "Date-based Forecast",
        "What-If Scenarios",
        "Feature Importance & Export"
    ])

    with tab1:
        st.subheader("Forecast Results (Last 15 Entries)")
        st.dataframe(processed_df[['Date', 'Sales', 'Forecasted_Sales']].tail(15))

        # Plot actual vs forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(processed_df['Date'], processed_df['Forecasted_Sales'], label='Forecasted Sales', linestyle='--', color='orange')
        if show_actuals:
            ax.plot(processed_df['Date'], processed_df['Sales'], label='Actual Sales', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.set_title('Actual vs Forecasted Sales')
        st.pyplot(fig)

    with tab2:
        st.subheader("Forecast for any date")

        future_max_date = dt.date(2030, 12, 31)

        selected_date = st.date_input(
            "Select a date to forecast sales for",
            value=min(processed_df['Date'].max().date() + dt.timedelta(days=30), future_max_date),
            min_value=processed_df['Date'].min().date(),
            max_value=future_max_date
        )
        st.write(f"Selected date: {selected_date}")

        input_features = build_features_for_date(selected_date, processed_df, default_stock, default_price)
        predicted_sales = rf_model.predict(input_features)[0]
        st.subheader(f"Predicted Sales on {selected_date}")
        st.write(f"{predicted_sales:.2f} units")

        actual_row = processed_df[processed_df['Date'] == pd.to_datetime(selected_date)]
        if not actual_row.empty:
            actual_sales = actual_row['Sales'].values[0]
            st.write(f"Actual Sales on that date: {actual_sales}")
        else:
            st.info("No actual sales data available for the selected date.")

        st.subheader("Forecast over a custom date range")

        start_date = st.date_input(
            "Forecast start date:",
            value=processed_df['Date'].min().date(),
            min_value=processed_df['Date'].min().date(),
            max_value=future_max_date
        )

        default_end = min(processed_df['Date'].max().date() + dt.timedelta(days=30), future_max_date)

        end_date = st.date_input(
            "Forecast end date:",
            value=default_end,
            min_value=start_date,
            max_value=future_max_date
        )

        st.write(f"Selected forecast period: {start_date} to {end_date}")

        if start_date > end_date:
            st.error("Start date must be before or equal to end date")
        else:
            all_dates = pd.date_range(start=start_date, end=end_date)
            st.write(f"Forecasting {len(all_dates)} days from {all_dates[0].date()} to {all_dates[-1].date()}")

            forecast_rows = []
            last_known_date = processed_df['Date'].max()
            last_known_features = processed_df.loc[processed_df['Date'] == last_known_date, features].iloc[0]

            for current_date in all_dates:
                if current_date in processed_df['Date'].values:
                    row = processed_df[processed_df['Date'] == current_date].iloc[0]
                    features_dict = {f: row[f] for f in features}
                else:
                    features_dict = {
                        'Month': current_date.month,
                        'DayOfWeek': current_date.weekday(),
                        'IsWeekend': int(current_date.weekday() >= 5),
                        'Sales_Lag7': 0,
                        'Sales_Lag30': 0,
                        'Sales_7d_Mean': 0,
                        'Sales_30d_Mean': 0,
                        'Stock': last_known_features['Stock'],
                        'Price': last_known_features['Price'],
                    }
                forecast_rows.append(features_dict)

            forecast_features_df = pd.DataFrame(forecast_rows)
            forecasted_sales = rf_model.predict(forecast_features_df)

            forecast_results = pd.DataFrame({
                'Date': all_dates,
                'Forecasted_Sales': forecasted_sales
            })

            st.dataframe(forecast_results)

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(forecast_results['Date'], forecast_results['Forecasted_Sales'], label='Forecasted Sales', color='purple')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Sales')
            ax2.set_title(f'Sales Forecast from {start_date} to {end_date}')
            ax2.legend()
            st.pyplot(fig2)

    with tab3:
        st.subheader("What-If Scenario Analysis")
        whatif_date = st.date_input(
            "Pick a date for What-If prediction",
            value=min(processed_df['Date'].max().date() + dt.timedelta(days=1), dt.date(2030, 12, 31)),
            min_value=processed_df['Date'].max().date(),
            max_value=dt.date(2030, 12, 31),
            key='whatif_date'
        )
        whatif_stock = st.number_input("Set Stock level:", min_value=0, value=default_stock, key='whatif_stock')
        whatif_price = st.number_input("Set Price:", min_value=0.0, value=default_price, format="%.2f", key='whatif_price')

        def build_custom_features(date, processed_df, stock_val, price_val):
            features_df = build_features_for_date(date, processed_df, default_stock=stock_val, default_price=price_val)
            features_df['Stock'] = stock_val
            features_df['Price'] = price_val
            return features_df

        custom_features = build_custom_features(whatif_date, processed_df, whatif_stock, whatif_price)
        custom_pred = rf_model.predict(custom_features)[0]
        st.info(f"Predicted sales for what-if scenario on {whatif_date}: {custom_pred:.2f} units")

    with tab4:
        st.subheader("Feature Importance")

        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            st.bar_chart(feat_imp_df.set_index('Feature'))
        else:
            st.write("Feature importance not available for this model.")

        # Download forecast for last interval if exists
        if 'forecast_results' in locals():
            csv = forecast_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Interval Forecast CSV",
                data=csv,
                file_name="forecast_interval.csv",
                mime="text/csv"
            )
        else:
            st.info("Upload data and select a forecast range to enable CSV download.")

        st.markdown("""
        ---
        **Notes**  
        - The forecasts use a pretrained Random Forest model.  
        - Modify Stock/Price in the What-If tab if you want to explore different scenarios.  
        - Feature importance explains which features most influence model predictions.
        """)

else:
    st.info("Please upload your sales data CSV file to get started.")
