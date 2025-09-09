"""
Fraud Prediction System
======================

This module implements ARIMA-based forecasting for fraud trends and patterns.
It provides model testing, selection, and prediction capabilities for the fraud chatbot.

Features:
- ARIMA model implementation with automatic parameter selection
- Model validation and performance testing
- Time series data preparation and feature engineering
- Confidence interval generation
- Model comparison and selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA forecasting will not work.")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will not work.")

import logging
logger = logging.getLogger(__name__)

class FraudPredictor:
    """
    ARIMA-based fraud prediction system with model testing and selection.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.validation_results = {}
        
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                target_column: str = 'is_fraud',
                                date_column: str = 'trans_date_trans_time',
                                frequency: str = 'D') -> pd.DataFrame:
        """
        Prepare time series data for forecasting.
        
        Args:
            df: Input DataFrame with transaction data
            target_column: Column to forecast (fraud rate, count, etc.)
            date_column: Date column name
            frequency: Resampling frequency ('D', 'W', 'M')
            
        Returns:
            Prepared time series DataFrame
        """
        try:
            # Convert date column
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Create time series data
            if target_column == 'is_fraud':
                # Calculate daily fraud rate
                ts_data = df.groupby(df[date_column].dt.date)[target_column].agg(['count', 'sum', 'mean']).reset_index()
                ts_data.columns = ['date', 'total_transactions', 'fraud_count', 'fraud_rate']
                ts_data['date'] = pd.to_datetime(ts_data['date'])
                ts_data = ts_data.set_index('date')
                
                # Resample to desired frequency
                ts_data = ts_data.resample(frequency).agg({
                    'total_transactions': 'sum',
                    'fraud_count': 'sum',
                    'fraud_rate': 'mean'
                })
                
                # Fill missing values
                ts_data = ts_data.fillna(method='ffill').fillna(0)
                
                return ts_data
            elif target_column == 'amt':
                # For fraud value forecasting - multiply amount by fraud flag
                ts_data = df.groupby(df[date_column].dt.date).apply(
                    lambda x: (x['amt'] * x['is_fraud']).sum()
                ).reset_index()
                ts_data.columns = ['date', 'fraud_value']
                ts_data['date'] = pd.to_datetime(ts_data['date'])
                ts_data = ts_data.set_index('date')
                
                # Resample to desired frequency
                ts_data = ts_data.resample(frequency).sum()
                
                # Fill missing values
                ts_data = ts_data.fillna(method='ffill').fillna(0)
                
                return ts_data
            else:
                # For other target columns
                ts_data = df.groupby(df[date_column].dt.date)[target_column].sum().reset_index()
                ts_data.columns = ['date', target_column]
                ts_data['date'] = pd.to_datetime(ts_data['date'])
                ts_data = ts_data.set_index('date')
                
                # Resample to desired frequency
                ts_data = ts_data.resample(frequency).sum()
                
                # Fill missing values
                ts_data = ts_data.fillna(method='ffill').fillna(0)
                
                return ts_data
                
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return pd.DataFrame()
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Test stationarity of time series using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        try:
            # Perform ADF test
            adf_result = adfuller(series.dropna())
            
            return {
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1],
                "critical_values": adf_result[4],
                "is_stationary": adf_result[1] < 0.05,
                "interpretation": "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
            }
        except Exception as e:
            logger.error(f"Error testing stationarity: {e}")
            return {"error": str(e)}
    
    def find_optimal_arima_params(self, series: pd.Series, 
                                 max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA parameters using grid search with AIC.
        
        Args:
            series: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)  # Default fallback
        
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        try:
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                                
                        except Exception:
                            continue
            
            logger.info(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
            return best_params
            
        except Exception as e:
            logger.error(f"Error finding optimal ARIMA params: {e}")
            return (1, 1, 1)
    
    def train_arima_model(self, series: pd.Series, 
                         order: Tuple[int, int, int] = None) -> Dict[str, Any]:
        """
        Train ARIMA model on time series data.
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q). If None, will find optimal parameters.
            
        Returns:
            Dictionary with model results and metrics
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        try:
            # Find optimal parameters if not provided
            if order is None:
                order = self.find_optimal_arima_params(series)
            
            # Train model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Calculate metrics
            aic = fitted_model.aic
            bic = fitted_model.bic
            log_likelihood = fitted_model.llf
            
            # Store model
            model_key = f"arima_{order[0]}_{order[1]}_{order[2]}"
            self.models[model_key] = fitted_model
            
            return {
                "model": fitted_model,
                "order": order,
                "aic": aic,
                "bic": bic,
                "log_likelihood": log_likelihood,
                "model_key": model_key,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {"error": str(e), "success": False}
    
    def generate_forecast(self, model, steps: int = 30, 
                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecast using trained ARIMA model.
        
        Args:
            model: Trained ARIMA model
            steps: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecast results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        try:
            # Generate forecast
            forecast = model.forecast(steps=steps)
            
            # Get confidence intervals
            conf_int = model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=model.data.dates[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1]
            })
            
            return {
                "forecast": forecast,
                "forecast_dates": forecast_dates,
                "confidence_interval": conf_int,
                "forecast_df": forecast_df,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {"error": str(e), "success": False}
    
    def validate_model(self, series: pd.Series, model, 
                      test_size: float = 0.2) -> Dict[str, Any]:
        """
        Validate ARIMA model using time series cross-validation.
        
        Args:
            series: Time series data
            model: Trained ARIMA model
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with validation metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        try:
            # Split data
            split_point = int(len(series) * (1 - test_size))
            train_data = series[:split_point]
            test_data = series[split_point:]
            
            # Retrain model on training data
            # Get the order from the model's spec attribute
            model_order = model.spec.order
            train_model = ARIMA(train_data, order=model_order)
            fitted_train_model = train_model.fit()
            
            # Generate forecast for test period
            forecast = fitted_train_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            mse = mean_squared_error(test_data, forecast)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (avoid division by zero)
            mape = np.mean(np.abs((test_data - forecast) / (test_data + 1e-8))) * 100
            
            return {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "test_size": len(test_data),
                "train_size": len(train_data),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {"error": str(e), "success": False}
    
    def test_multiple_models(self, series: pd.Series, 
                           model_orders: List[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Test multiple ARIMA models and select the best one.
        
        Args:
            series: Time series data
            model_orders: List of (p, d, q) orders to test
            
        Returns:
            Dictionary with model comparison results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        if model_orders is None:
            model_orders = [
                (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
                (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2),
                (0, 1, 1), (0, 1, 2), (1, 0, 1), (1, 0, 2)
            ]
        
        results = {}
        best_model = None
        best_aic = float('inf')
        
        try:
            for order in model_orders:
                try:
                    # Train model
                    model_result = self.train_arima_model(series, order)
                    
                    if model_result.get("success", False):
                        # Validate model
                        validation = self.validate_model(series, model_result["model"])
                        
                        # Store results
                        results[f"arima_{order[0]}_{order[1]}_{order[2]}"] = {
                            "order": order,
                            "aic": model_result["aic"],
                            "bic": model_result["bic"],
                            "validation": validation,
                            "model": model_result["model"]
                        }
                        
                        # Check if this is the best model
                        if model_result["aic"] < best_aic:
                            best_aic = model_result["aic"]
                            best_model = model_result["model"]
                            self.best_model = best_model
                            self.best_params = order
                        
                except Exception as e:
                    logger.warning(f"Failed to train model {order}: {e}")
                    continue
            
            # Sort results by AIC
            sorted_results = sorted(results.items(), key=lambda x: x[1]["aic"])
            
            return {
                "results": dict(sorted_results),
                "best_model": best_model,
                "best_params": self.best_params,
                "best_aic": best_aic,
                "total_models_tested": len(results),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error testing multiple models: {e}")
            return {"error": str(e), "success": False}
    
    def predict_fraud_trends(self, df: pd.DataFrame, 
                           forecast_days: int = 30,
                           target_column: str = 'is_fraud') -> Dict[str, Any]:
        """
        Main method to predict fraud trends using ARIMA.
        
        Args:
            df: Transaction DataFrame
            forecast_days: Number of days to forecast
            target_column: Column to forecast
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare time series data
            ts_data = self.prepare_time_series_data(df, target_column)
            
            if ts_data.empty:
                return {"error": "Failed to prepare time series data"}
            
            # Determine the correct column for analysis
            if target_column == 'is_fraud':
                analysis_column = 'fraud_rate'
            elif target_column == 'amt':
                analysis_column = 'fraud_value'
            else:
                analysis_column = target_column
            
            # Test stationarity
            stationarity = self.test_stationarity(ts_data[analysis_column])
            
            # Test multiple models
            model_comparison = self.test_multiple_models(ts_data[analysis_column])
            
            if not model_comparison.get("success", False):
                return {"error": "Failed to test models"}
            
            # Generate forecast with best model
            best_model = model_comparison["best_model"]
            forecast_result = self.generate_forecast(best_model, forecast_days)
            
            if not forecast_result.get("success", False):
                return {"error": "Failed to generate forecast"}
            
            # Prepare response
            response = {
                "success": True,
                "forecast_days": forecast_days,
                "model_used": f"ARIMA{model_comparison['best_params']}",
                "model_aic": model_comparison["best_aic"],
                "stationarity_test": stationarity,
                "model_comparison": model_comparison["results"],
                "forecast_data": forecast_result["forecast_df"].to_dict('records'),
                "forecast_summary": {
                    "mean_forecast": float(forecast_result["forecast"].mean()),
                    "forecast_std": float(forecast_result["forecast"].std()),
                    "forecast_range": [
                        float(forecast_result["forecast"].min()),
                        float(forecast_result["forecast"].max())
                    ]
                },
                "historical_data": ts_data.tail(30).to_dict('records')
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error predicting fraud trends: {e}")
            return {"error": str(e), "success": False}
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of model performance and validation results.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        try:
            # Calculate average metrics across all models
            all_mae = [result["validation"]["mae"] for result in self.validation_results.values() 
                      if result["validation"].get("success", False)]
            all_rmse = [result["validation"]["rmse"] for result in self.validation_results.values() 
                       if result["validation"].get("success", False)]
            all_mape = [result["validation"]["mape"] for result in self.validation_results.values() 
                       if result["validation"].get("success", False)]
            
            return {
                "total_models_tested": len(self.validation_results),
                "successful_models": len([r for r in self.validation_results.values() 
                                        if r["validation"].get("success", False)]),
                "average_mae": np.mean(all_mae) if all_mae else 0,
                "average_rmse": np.mean(all_rmse) if all_rmse else 0,
                "average_mape": np.mean(all_mape) if all_mape else 0,
                "best_model": self.best_params,
                "best_aic": self.best_model.aic if self.best_model else None
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the predictor
    predictor = FraudPredictor()
    print("Fraud Predictor initialized successfully!")
    print(f"Statsmodels available: {STATSMODELS_AVAILABLE}")
    print(f"Scikit-learn available: {SKLEARN_AVAILABLE}")
