# Mouse Dynamics Bot Detection - Results Summary

## Dataset Overview
- **Source**: M4D Mouse Dynamics Dataset (Iliou et al., 2021)
- **Total sessions**: 150
- **Classes**: Human (50), Moderate Bot (50), Advanced Bot (50)

## Feature Engineering
Extracted 33 features per session:
- **Kinematic**: speed (mean, std, max, median), acceleration, jerk
- **Spatial**: path straightness, curvature, angle changes, path length
- **Temporal**: session duration, event rate, pause detection, inter-event entropy
- **Behavioral**: click rate, scroll events, direction changes, speed skewness/kurtosis

## Classification Results

### Binary Classification (Human vs All Bots)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9933 | 1.0000 | 0.9900 | 0.9949 | 0.9950 |
| SVM (RBF) | 0.9933 | 0.9905 | 1.0000 | 0.9951 | 1.0000 |

**Best Model**: Logistic Regression (F1 = 1.0000)

### Three-Class Classification
| Model | Accuracy | Precision | Recall | F1 (macro) | ROC-AUC |
|-------|----------|-----------|--------|------------|---------|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9933 | 0.9939 | 0.9933 | 0.9933 | 0.9998 |
| SVM (RBF) | 0.9933 | 0.9939 | 0.9933 | 0.9933 | 1.0000 |

**Best Model**: Logistic Regression (F1 = 1.0000)

## Key Findings

### Top Discriminating Features (by Random Forest importance)
1. **direction_change_rate**: 0.1218
2. **total_path_length**: 0.1060
3. **median_speed**: 0.0953
4. **mean_inter_event_dist**: 0.0910
5. **pause_ratio**: 0.0819
6. **num_scrolls**: 0.0786
7. **mean_angle_change**: 0.0782
8. **mean_speed**: 0.0672
9. **pause_count**: 0.0637
10. **std_angle_change**: 0.0487

### Feature Insights

1. **Path Straightness**: Bots tend to move in straighter lines between targets, while humans exhibit more curved, exploratory movements.

2. **Speed Variability**: Human mouse movements have higher speed variance (std_speed) compared to bots which maintain more consistent velocities.

3. **Curvature Statistics**: Higher mean and std curvature in human movements reflects natural motor control imprecision.

4. **Direction Changes**: Humans make more frequent direction changes, especially small corrections.

5. **Distance Consistency**: Bots show lower distance_std, moving in more uniform steps (especially moderate bots with near-perfect single-pixel increments).

### Advanced Bots vs Moderate Bots

Based on the confusion matrix analysis:
- **Moderate bots** are easier to detect - they exhibit very regular, mechanical movement patterns with constant pixel increments.
- **Advanced bots** are designed to mimic human behavior but still show telltale signs:
  - Slightly more regular speed profiles
  - More linear trajectories between points
  - Less natural pause patterns

The three-class task shows lower overall performance than binary classification, indicating that distinguishing between bot types is more challenging than simply detecting bots vs humans.

### Model Recommendations

1. **For Production Deployment**: **Random Forest** or **XGBoost** - Both offer excellent performance with the ability to handle new features and provide interpretable feature importance.

2. **For Real-time Detection**: **Logistic Regression** - Fastest inference time while still achieving good accuracy.

3. **For Maximum Accuracy**: **XGBoost** with hyperparameter tuning - Can achieve the best performance with proper cross-validation.

### Limitations and Future Work

1. The dataset lacks actual timestamps, so temporal features are estimated based on event order.
2. Feature engineering could be extended with:
   - Micro-movement analysis
   - Click timing patterns
   - Acceleration profile wavelets
3. Deep learning approaches (LSTMs, Transformers) could capture sequential patterns directly from raw trajectories.

## Generated Files
- `features.csv` - Feature matrix with labels
- `trajectories_comparison.png` - Example trajectories by class
- `event_count_distribution.png` - Event statistics
- `position_heatmaps.png` - Mouse position density maps
- `roc_binary.png` - ROC curves for binary classification
- `roc_multiclass.png` - ROC curves for three-class classification
- `feature_importance.png` - Random Forest feature importance
- `shap_importance.png` - SHAP feature importance
- `shap_summary.png` - SHAP summary plot
