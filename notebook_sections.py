# This file contains the remaining notebook sections as reference
# Copy these into notebook cells as needed

# SECTION 5: REGRESSION MODELS (Practical 3, CO2)
regression_section = """
## Section 5: Regression Models
### Practical 3 | CO2: Simple, Multiple, Polynomial, Logistic Regression

Apply various regression techniques and evaluate using MAE, MSE, RMSE, R².
"""

regression_code_1 = """
# Initialize results storage
regression_results = {}

print("="*80)
print("REGRESSION MODELS - PRACTICAL 3 (CO2)")
print("="*80)

# Model 1: Simple Linear Regression
# Predict 'time_in_hospital' as a continuous target (example)
print("\\n1. SIMPLE LINEAR REGRESSION")
print("-"*80)

# Use a single feature
X_simple = X_train_scaled[['num_medications']].values
y_continuous = df_encoded.loc[X_train.index, 'time_in_hospital'].values

lr_simple = LinearRegression()
lr_simple.fit(X_simple, y_continuous)

X_simple_test = X_test_scaled[['num_medications']].values
y_continuous_test = df_encoded.loc[X_test.index, 'time_in_hospital'].values

y_pred_simple = lr_simple.predict(X_simple_test)

mae_simple = mean_absolute_error(y_continuous_test, y_pred_simple)
mse_simple = mean_squared_error(y_continuous_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_continuous_test, y_pred_simple)

print(f"MAE: {mae_simple:.4f}")
print(f"MSE: {mse_simple:.4f}")
print(f"RMSE: {rmse_simple:.4f}")
print(f"R² Score: {r2_simple:.4f}")

regression_results['Simple LR'] = {
    'MAE': mae_simple, 'MSE': mse_simple, 'RMSE': rmse_simple, 'R²': r2_simple
}
"""

regression_code_2 = """
# Model 2: Multiple Linear Regression
print("\\n2. MULTIPLE LINEAR REGRESSION")
print("-"*80)

lr_multiple = LinearRegression()
lr_multiple.fit(X_train_scaled, y_continuous)
y_pred_multiple = lr_multiple.predict(X_test_scaled)

mae_multiple = mean_absolute_error(y_continuous_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_continuous_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_continuous_test, y_pred_multiple)

print(f"MAE: {mae_multiple:.4f}")
print(f"MSE: {mse_multiple:.4f}")
print(f"RMSE: {rmse_multiple:.4f}")
print(f"R² Score: {r2_multiple:.4f}")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_multiple.coef_
}).sort_values('Coefficient', ascending=False, key=abs)

print(f"\\nTop 10 Features by Coefficient Magnitude:")
print(feature_importance.head(10))

regression_results['Multiple LR'] = {
    'MAE': mae_multiple, 'MSE': mse_multiple, 'RMSE': rmse_multiple, 'R²': r2_multiple
}
"""

regression_code_3 = """
# Model 3: Polynomial Regression
print("\\n3. POLYNOMIAL REGRESSION (Degree 2)")
print("-"*80)

# Use subset of features to avoid dimensionality explosion
poly_features = ['num_medications', 'time_in_hospital', 'num_lab_procedures', 'number_inpatient']
X_train_poly = X_train_scaled[poly_features]
X_test_poly = X_test_scaled[poly_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_transformed = poly.fit_transform(X_train_poly)
X_test_poly_transformed = poly.transform(X_test_poly)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly_transformed, y_continuous)
y_pred_poly = lr_poly.predict(X_test_poly_transformed)

mae_poly = mean_absolute_error(y_continuous_test, y_pred_poly)
mse_poly = mean_squared_error(y_continuous_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_continuous_test, y_pred_poly)

print(f"MAE: {mae_poly:.4f}")
print(f"MSE: {mse_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")
print(f"R² Score: {r2_poly:.4f}")
print(f"Polynomial features created: {X_train_poly_transformed.shape[1]}")

regression_results['Polynomial LR'] = {
    'MAE': mae_poly, 'MSE': mse_poly, 'RMSE': rmse_poly, 'R²': r2_poly
}
"""

regression_code_4 = """
# Model 4: Logistic Regression (for binary classification)
print("\\n4. LOGISTIC REGRESSION (Binary Classification)")
print("-"*80)

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]

accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_prob_log)

print(f"Accuracy: {accuracy_log:.4f}")
print(f"Precision: {precision_log:.4f}")
print(f"Recall: {recall_log:.4f}")
print(f"F1-Score: {f1_log:.4f}")
print(f"AUC-ROC: {auc_log:.4f}")

print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=['Not Readmitted', 'Readmitted <30d']))

regression_results['Logistic Regression'] = {
    'Accuracy': accuracy_log, 'Precision': precision_log, 'Recall': recall_log,
    'F1': f1_log, 'AUC': auc_log
}

# Store for later use
models_performance = {'Logistic Regression': regression_results['Logistic Regression']}

print("\\n✅ Regression models complete!")
"""

# Save results visualization
regression_viz = """
# Regression Results Summary
print("\\n" + "="*80)
print("REGRESSION MODELS SUMMARY")
print("="*80)

regression_df = pd.DataFrame(regression_results).T
print(regression_df)

# Visualize regression results
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: R² scores
r2_scores = regression_df['R²'].dropna()
ax[0].barh(r2_scores.index, r2_scores.values, color='steelblue')
ax[0].set_xlabel('R² Score')
ax[0].set_title('R² Scores - Regression Models', fontweight='bold')
ax[0].axvline(x=0, color='red', linestyle='--', alpha=0.3)

# Plot 2: RMSE values
rmse_scores = regression_df['RMSE'].dropna()
ax[1].barh(rmse_scores.index, rmse_scores.values, color='coral')
ax[1].set_xlabel('RMSE')
ax[1].set_title('RMSE - Regression Models (Lower is Better)', fontweight='bold')

plt.tight_layout()
plt.show()
"""

print("Reference file created successfully!")
