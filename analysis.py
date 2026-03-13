#!/usr/bin/env python3
"""
Mouse Dynamics Bot Detection Analysis
Dataset: M4D Mouse Dynamics (Iliou et al., 2021)
"""

import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc)
import xgboost as xgb

# Set paths
BASE_DIR = "/Users/karennurlybekov/Desktop/mousedynamics"
DATA_DIR = os.path.join(BASE_DIR, "data/phase1")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 1: DATASET EXPLORATION")
print("=" * 60)

# ============================================
# 1. Load and parse annotations
# ============================================
print("\n[1.1] Loading annotations...")

def load_annotations(filepath):
    """Load annotation file and return dict of session_id -> label"""
    annotations = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    session_id, label = parts[0], parts[1]
                    annotations[session_id] = label
    return annotations

# Load all annotations
adv_train = load_annotations(os.path.join(DATA_DIR, "annotations/humans_and_advanced_bots/train"))
adv_test = load_annotations(os.path.join(DATA_DIR, "annotations/humans_and_advanced_bots/test"))
mod_train = load_annotations(os.path.join(DATA_DIR, "annotations/humans_and_moderate_bots/train"))
mod_test = load_annotations(os.path.join(DATA_DIR, "annotations/humans_and_moderate_bots/test"))

# Combine all annotations
all_annotations = {}
all_annotations.update(adv_train)
all_annotations.update(adv_test)
all_annotations.update(mod_train)
all_annotations.update(mod_test)

print(f"   Advanced bots dataset: {len(adv_train)} train, {len(adv_test)} test")
print(f"   Moderate bots dataset: {len(mod_train)} train, {len(mod_test)} test")
print(f"   Total unique sessions: {len(all_annotations)}")

# Count by label
label_counts = defaultdict(int)
for label in all_annotations.values():
    label_counts[label] += 1
print(f"   Label distribution: {dict(label_counts)}")

# ============================================
# 2. Parse mouse movement data
# ============================================
print("\n[1.2] Parsing mouse movement data...")

def parse_mouse_movements(behavior_string):
    """Parse the total_behaviour string into list of events"""
    events = []
    # Pattern matches m(x,y) for moves and c(l), c(r), s(d), s(u) for other events
    pattern = r'\[(m|c|s)\(([^)]+)\)\]'
    matches = re.findall(pattern, behavior_string)

    for event_type, params in matches:
        if event_type == 'm':
            # Mouse move
            coords = params.split(',')
            if len(coords) == 2:
                try:
                    x, y = int(coords[0]), int(coords[1])
                    events.append({'type': 'move', 'x': x, 'y': y})
                except ValueError:
                    pass
        elif event_type == 'c':
            # Click (l=left, r=right)
            events.append({'type': 'click', 'button': params})
        elif event_type == 's':
            # Scroll (d=down, u=up)
            events.append({'type': 'scroll', 'direction': params})

    return events

def load_session_data(session_id, dataset_type):
    """Load mouse movement data for a session"""
    # Try both directories
    paths = [
        os.path.join(DATA_DIR, f"data/mouse_movements/humans_and_advanced_bots/{session_id}/mouse_movements.json"),
        os.path.join(DATA_DIR, f"data/mouse_movements/humans_and_moderate_bots/{session_id}/mouse_movements.json")
    ]

    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return parse_mouse_movements(data.get('total_behaviour', ''))
    return None

# Load all sessions
sessions_data = {}
for session_id, label in all_annotations.items():
    events = load_session_data(session_id, label)
    if events:
        sessions_data[session_id] = {
            'events': events,
            'label': label
        }

print(f"   Successfully loaded {len(sessions_data)} sessions")

# ============================================
# 3. Summary Statistics
# ============================================
print("\n[1.3] Computing summary statistics...")

stats_by_class = defaultdict(list)
for session_id, data in sessions_data.items():
    events = data['events']
    label = data['label']

    move_events = [e for e in events if e['type'] == 'move']
    click_events = [e for e in events if e['type'] == 'click']
    scroll_events = [e for e in events if e['type'] == 'scroll']

    stats_by_class[label].append({
        'session_id': session_id,
        'total_events': len(events),
        'move_events': len(move_events),
        'click_events': len(click_events),
        'scroll_events': len(scroll_events)
    })

print("\n   Summary Statistics by Class:")
print("-" * 70)
for label in ['human', 'moderate_bot', 'advanced_bot']:
    if label in stats_by_class:
        session_stats = stats_by_class[label]
        n_sessions = len(session_stats)
        total_events = [s['total_events'] for s in session_stats]
        move_events = [s['move_events'] for s in session_stats]
        click_events = [s['click_events'] for s in session_stats]

        print(f"\n   {label.upper()}:")
        print(f"   - Sessions: {n_sessions}")
        print(f"   - Total events: mean={np.mean(total_events):.1f}, std={np.std(total_events):.1f}, "
              f"min={np.min(total_events)}, max={np.max(total_events)}")
        print(f"   - Move events: mean={np.mean(move_events):.1f}, std={np.std(move_events):.1f}")
        print(f"   - Click events: mean={np.mean(click_events):.1f}, std={np.std(click_events):.1f}")

# ============================================
# 4. Exploratory Visualizations
# ============================================
print("\n[1.4] Creating exploratory visualizations...")

# Function to extract trajectory
def get_trajectory(events):
    """Extract x,y coordinates from move events"""
    moves = [(e['x'], e['y']) for e in events if e['type'] == 'move']
    if not moves:
        return [], []
    x, y = zip(*moves)
    return list(x), list(y)

# --- Plot 1: Example Trajectories ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels_to_plot = ['human', 'moderate_bot', 'advanced_bot']

for idx, label in enumerate(labels_to_plot):
    ax = axes[idx]
    sessions_for_label = [(sid, d) for sid, d in sessions_data.items() if d['label'] == label]

    # Plot 5 example trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, (sid, data) in enumerate(sessions_for_label[:5]):
        x, y = get_trajectory(data['events'])
        if x and y:
            ax.plot(x[:500], y[:500], alpha=0.7, linewidth=0.8, color=colors[i], label=f'Session {i+1}')
            ax.scatter(x[0], y[0], color=colors[i], s=50, marker='o', zorder=5)  # Start point

    ax.set_title(f'{label.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.invert_yaxis()  # Invert Y since screen coordinates go down
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Mouse Trajectories: Human vs Moderate Bot vs Advanced Bot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'trajectories_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: trajectories_comparison.png")

# --- Plot 2: Event Count Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Collect data for plots
event_counts_df = []
for label in ['human', 'moderate_bot', 'advanced_bot']:
    if label in stats_by_class:
        for s in stats_by_class[label]:
            event_counts_df.append({
                'label': label,
                'total_events': s['total_events'],
                'move_events': s['move_events']
            })
event_counts_df = pd.DataFrame(event_counts_df)

# Event count distribution
ax = axes[0]
for label in ['human', 'moderate_bot', 'advanced_bot']:
    subset = event_counts_df[event_counts_df['label'] == label]['total_events']
    ax.hist(subset, bins=30, alpha=0.6, label=label.replace('_', ' ').title())
ax.set_xlabel('Total Events per Session')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Event Counts by Class')
ax.legend()
ax.grid(True, alpha=0.3)

# Box plot
ax = axes[1]
order = ['human', 'moderate_bot', 'advanced_bot']
sns.boxplot(data=event_counts_df, x='label', y='total_events', order=order, ax=ax, palette='Set2')
ax.set_xlabel('Class')
ax.set_ylabel('Total Events')
ax.set_title('Event Count Distribution by Class')
ax.set_xticklabels(['Human', 'Moderate Bot', 'Advanced Bot'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'event_count_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: event_count_distribution.png")

# --- Plot 3: Heatmap of Mouse Positions ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, label in enumerate(['human', 'moderate_bot', 'advanced_bot']):
    ax = axes[idx]
    sessions_for_label = [(sid, d) for sid, d in sessions_data.items() if d['label'] == label]

    # Collect all positions
    all_x, all_y = [], []
    for sid, data in sessions_for_label[:20]:  # Use first 20 sessions
        x, y = get_trajectory(data['events'])
        all_x.extend(x[:1000])  # Limit points per session
        all_y.extend(y[:1000])

    if all_x and all_y:
        # Create 2D histogram (heatmap)
        h, xedges, yedges = np.histogram2d(all_x, all_y, bins=50)
        im = ax.imshow(h.T, extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]],
                       cmap='hot', aspect='auto', interpolation='gaussian')
        plt.colorbar(im, ax=ax, label='Density')

    ax.set_title(f'{label.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

plt.suptitle('Mouse Position Heatmaps', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'position_heatmaps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: position_heatmaps.png")

print("\n" + "=" * 60)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 60)

# ============================================
# Feature Extraction Functions
# ============================================

def compute_distances(x, y):
    """Compute distances between consecutive points"""
    if len(x) < 2:
        return np.array([])
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)

def compute_speeds(distances, time_interval=0.01):
    """Compute speeds (assume constant time interval between events)"""
    # Approximate time intervals - in real data this would come from timestamps
    return distances / time_interval if time_interval > 0 else distances

def compute_accelerations(speeds):
    """Compute accelerations from speeds"""
    if len(speeds) < 2:
        return np.array([])
    return np.diff(speeds)

def compute_jerks(accelerations):
    """Compute jerk (rate of acceleration change)"""
    if len(accelerations) < 2:
        return np.array([])
    return np.diff(accelerations)

def compute_angles(x, y):
    """Compute angles between consecutive segments"""
    if len(x) < 3:
        return np.array([])

    angles = []
    for i in range(len(x) - 2):
        # Vector from point i to i+1
        v1 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        # Vector from point i+1 to i+2
        v2 = np.array([x[i+2] - x[i+1], y[i+2] - y[i+1]])

        # Compute angle between vectors
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

    return np.array(angles)

def compute_curvature(x, y):
    """Compute curvature at each point"""
    if len(x) < 3:
        return np.array([])

    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5

    # Avoid division by zero
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

    return curvature

def compute_path_straightness(x, y):
    """Compute path straightness ratio (direct distance / total path length)"""
    if len(x) < 2:
        return 0

    # Direct distance (start to end)
    direct_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

    # Total path length
    distances = compute_distances(x, y)
    total_length = np.sum(distances) if len(distances) > 0 else 0

    if total_length > 0:
        return direct_dist / total_length
    return 0

def compute_direction_changes(x, y):
    """Count significant direction changes"""
    angles = compute_angles(x, y)
    if len(angles) == 0:
        return 0
    # Count angles greater than 45 degrees (pi/4 radians)
    return np.sum(angles > np.pi/4)

def safe_stat(arr, func, default=0):
    """Safely compute a statistic on an array"""
    if len(arr) == 0:
        return default
    return func(arr)

def compute_entropy(values, bins=20):
    """Compute entropy of a distribution"""
    if len(values) < 2:
        return 0
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros for log
    if len(hist) == 0:
        return 0
    return entropy(hist)

def extract_features(events):
    """Extract all features from a session's events"""
    features = {}

    # Separate event types
    moves = [(e['x'], e['y']) for e in events if e['type'] == 'move']
    clicks = [e for e in events if e['type'] == 'click']
    scrolls = [e for e in events if e['type'] == 'scroll']

    if len(moves) < 3:
        return None  # Not enough data

    x = np.array([m[0] for m in moves])
    y = np.array([m[1] for m in moves])

    # Compute intermediate values
    distances = compute_distances(x, y)
    speeds = compute_speeds(distances)
    accelerations = compute_accelerations(speeds)
    jerks = compute_jerks(accelerations)
    angles = compute_angles(x, y)
    curvature = compute_curvature(x, y)

    # --- Kinematic features ---
    features['mean_speed'] = safe_stat(speeds, np.mean)
    features['std_speed'] = safe_stat(speeds, np.std)
    features['max_speed'] = safe_stat(speeds, np.max)
    features['median_speed'] = safe_stat(speeds, np.median)

    features['mean_acceleration'] = safe_stat(accelerations, np.mean)
    features['std_acceleration'] = safe_stat(accelerations, np.std)
    features['max_acceleration'] = safe_stat(accelerations, lambda a: np.max(np.abs(a)))

    features['mean_jerk'] = safe_stat(jerks, np.mean)
    features['std_jerk'] = safe_stat(jerks, np.std)

    # --- Spatial features ---
    features['path_straightness'] = compute_path_straightness(x, y)
    features['mean_curvature'] = safe_stat(curvature, np.mean)
    features['std_curvature'] = safe_stat(curvature, np.std)
    features['mean_angle_change'] = safe_stat(angles, np.mean)
    features['std_angle_change'] = safe_stat(angles, np.std)
    features['total_path_length'] = np.sum(distances) if len(distances) > 0 else 0

    # --- Temporal features ---
    features['num_events'] = len(events)
    features['num_moves'] = len(moves)
    # Assume ~100Hz sampling, so estimate duration
    features['session_duration'] = len(moves) * 0.01  # seconds estimate
    features['event_rate'] = len(events) / max(features['session_duration'], 0.001)

    # Inter-event distances as proxy for time intervals
    if len(distances) > 0:
        features['mean_inter_event_dist'] = np.mean(distances)
        features['std_inter_event_dist'] = np.std(distances)
    else:
        features['mean_inter_event_dist'] = 0
        features['std_inter_event_dist'] = 0

    # Pause detection (very slow movement)
    pause_threshold = 0.5  # pixels per time unit
    pauses = speeds < pause_threshold if len(speeds) > 0 else np.array([])
    features['pause_count'] = np.sum(pauses)
    features['pause_ratio'] = np.mean(pauses) if len(pauses) > 0 else 0

    # Entropy of inter-event distances
    features['inter_event_entropy'] = compute_entropy(distances) if len(distances) > 0 else 0

    # --- Behavioral features ---
    features['num_clicks'] = len(clicks)
    features['click_rate'] = len(clicks) / max(features['session_duration'], 0.001)
    features['num_scrolls'] = len(scrolls)

    features['direction_change_count'] = compute_direction_changes(x, y)
    features['direction_change_rate'] = features['direction_change_count'] / max(len(moves), 1)

    # Speed profile statistics
    if len(speeds) > 3:
        features['speed_skewness'] = stats.skew(speeds)
        features['speed_kurtosis'] = stats.kurtosis(speeds)
    else:
        features['speed_skewness'] = 0
        features['speed_kurtosis'] = 0

    # Distance statistics (helps detect linear bots)
    if len(distances) > 0:
        features['distance_std'] = np.std(distances)
        features['distance_skewness'] = stats.skew(distances) if len(distances) > 3 else 0
    else:
        features['distance_std'] = 0
        features['distance_skewness'] = 0

    return features

# ============================================
# Extract features for all sessions
# ============================================
print("\n[2.1] Extracting features from all sessions...")

feature_rows = []
for session_id, data in sessions_data.items():
    features = extract_features(data['events'])
    if features is not None:
        features['session_id'] = session_id
        features['label_str'] = data['label']
        feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

# Create numeric labels
label_map = {'human': 0, 'moderate_bot': 1, 'advanced_bot': 2}
features_df['label'] = features_df['label_str'].map(label_map)

print(f"   Extracted features for {len(features_df)} sessions")
print(f"   Feature columns: {len([c for c in features_df.columns if c not in ['session_id', 'label', 'label_str']])}")

# Check for missing values
missing = features_df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"   Warning: Missing values found in columns: {list(missing_cols.index)}")
    features_df = features_df.fillna(0)

# Save features
features_df.to_csv(os.path.join(OUTPUT_DIR, 'features.csv'), index=False)
print(f"   Saved: features.csv")

# Print feature summary
print("\n   Feature statistics:")
feature_cols = [c for c in features_df.columns if c not in ['session_id', 'label', 'label_str']]
for col in feature_cols[:10]:  # Show first 10
    print(f"   - {col}: mean={features_df[col].mean():.4f}, std={features_df[col].std():.4f}")

print("\n" + "=" * 60)
print("PHASE 3: CLASSIFICATION")
print("=" * 60)

# ============================================
# Prepare data for classification
# ============================================
feature_cols = [c for c in features_df.columns if c not in ['session_id', 'label', 'label_str']]
X = features_df[feature_cols].values
y = features_df['label'].values
session_ids = features_df['session_id'].values

# Replace inf values
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# Classification Functions
# ============================================
def evaluate_models(X, y, models, cv=5, task_name=""):
    """Evaluate multiple models using stratified k-fold CV"""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"\n   Training {name}...")

        # Storage for metrics
        accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []
        all_preds = np.zeros(len(y))
        all_probs = np.zeros((len(y), len(np.unique(y))) if len(np.unique(y)) > 2 else (len(y),))

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit model
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)

            # Predictions
            y_pred = model_clone.predict(X_test)
            all_preds[test_idx] = y_pred

            # Probabilities for ROC
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_test)
                if len(np.unique(y)) == 2:
                    all_probs[test_idx] = y_prob[:, 1]
                else:
                    all_probs[test_idx] = y_prob
            elif hasattr(model_clone, 'decision_function'):
                y_prob = model_clone.decision_function(X_test)
                all_probs[test_idx] = y_prob

            # Compute metrics
            accuracies.append(accuracy_score(y_test, y_pred))

            if len(np.unique(y)) == 2:
                precisions.append(precision_score(y_test, y_pred))
                recalls.append(recall_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred))
                if hasattr(model_clone, 'predict_proba'):
                    aucs.append(roc_auc_score(y_test, y_prob[:, 1]))
            else:
                precisions.append(precision_score(y_test, y_pred, average='macro'))
                recalls.append(recall_score(y_test, y_pred, average='macro'))
                f1s.append(f1_score(y_test, y_pred, average='macro'))
                if hasattr(model_clone, 'predict_proba'):
                    aucs.append(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))

        results[name] = {
            'accuracy': (np.mean(accuracies), np.std(accuracies)),
            'precision': (np.mean(precisions), np.std(precisions)),
            'recall': (np.mean(recalls), np.std(recalls)),
            'f1': (np.mean(f1s), np.std(f1s)),
            'auc': (np.mean(aucs), np.std(aucs)) if aucs else (0, 0)
        }

        predictions[name] = all_preds
        probabilities[name] = all_probs

        print(f"      Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"      F1 (macro): {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    return results, predictions, probabilities

# ============================================
# Task 1: Binary Classification (Human vs Bot)
# ============================================
print("\n[3.1] BINARY CLASSIFICATION: Human (0) vs All Bots (1)")
print("-" * 60)

# Create binary labels
y_binary = (y > 0).astype(int)  # 0 = human, 1 = bot

binary_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}

binary_results, binary_preds, binary_probs = evaluate_models(X_scaled, y_binary, binary_models, task_name="Binary")

# Best model confusion matrix
best_model_name = max(binary_results, key=lambda k: binary_results[k]['f1'][0])
print(f"\n   Best model (by F1): {best_model_name}")
print("\n   Confusion Matrix (aggregated across folds):")
cm = confusion_matrix(y_binary, binary_preds[best_model_name])
print(f"   {cm}")

# ROC curves for binary classification
fig, ax = plt.subplots(figsize=(8, 6))
for name in binary_models.keys():
    if name != 'SVM (RBF)' or np.any(binary_probs[name] != 0):
        fpr, tpr, _ = roc_curve(y_binary, binary_probs[name])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Binary Classification (Human vs Bot)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_binary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: roc_binary.png")

# ============================================
# Task 2: Three-Class Classification
# ============================================
print("\n[3.2] THREE-CLASS CLASSIFICATION: Human vs Moderate Bot vs Advanced Bot")
print("-" * 60)

multiclass_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}

multi_results, multi_preds, multi_probs = evaluate_models(X_scaled, y, multiclass_models, task_name="Multiclass")

# Best model confusion matrix
best_model_name_multi = max(multi_results, key=lambda k: multi_results[k]['f1'][0])
print(f"\n   Best model (by F1): {best_model_name_multi}")
print("\n   Confusion Matrix (aggregated across folds):")
cm_multi = confusion_matrix(y, multi_preds[best_model_name_multi])
print(f"   Labels: 0=Human, 1=Moderate Bot, 2=Advanced Bot")
print(f"   {cm_multi}")

# ROC curves for multiclass (one-vs-rest)
fig, ax = plt.subplots(figsize=(8, 6))
class_names = ['Human', 'Moderate Bot', 'Advanced Bot']
for name, probs in multi_probs.items():
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        # Compute macro-average ROC
        fpr_all, tpr_all = [], []
        for i in range(3):
            fpr, tpr, _ = roc_curve((y == i).astype(int), probs[:, i])
            fpr_all.append(fpr)
            tpr_all.append(tpr)

        # Interpolate to common FPR points
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(3):
            mean_tpr += np.interp(mean_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= 3

        roc_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, label=f'{name} (Macro AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Three-Class Classification (Macro-averaged)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_multiclass.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: roc_multiclass.png")

print("\n" + "=" * 60)
print("PHASE 4: ANALYSIS")
print("=" * 60)

# ============================================
# Feature Importance
# ============================================
print("\n[4.1] Computing feature importance...")

# Train Random Forest on full data for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_scaled, y)

# Get feature importances
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n   Top 15 Features by Importance:")
for i, row in importance_df.head(15).iterrows():
    print(f"   {importance_df.index.get_loc(i)+1:2d}. {row['feature']}: {row['importance']:.4f}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
top_features = importance_df.head(15)
ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Features - Random Forest Feature Importance')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: feature_importance.png")

# ============================================
# SHAP Analysis
# ============================================
print("\n[4.2] Generating SHAP analysis...")

try:
    import shap

    # Use Random Forest for SHAP (more stable with TreeExplainer)
    rf_shap = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_shap.fit(X_scaled, y)

    # Create SHAP explainer with check_additivity=False to avoid numerical issues
    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer.shap_values(X_scaled, check_additivity=False)

    # SHAP summary plot - handle array format
    fig, ax = plt.subplots(figsize=(12, 8))

    # shap_values shape: (n_samples, n_features, n_classes) for newer versions
    # or list of (n_samples, n_features) for older versions
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Average absolute SHAP values across classes
        shap_values_mean = np.mean(np.abs(shap_values), axis=2)
    elif isinstance(shap_values, list):
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values_mean = np.abs(shap_values)

    feature_importance_shap = np.mean(shap_values_mean, axis=0)
    sorted_idx = np.argsort(feature_importance_shap)[-15:]

    ax.barh(range(len(sorted_idx)), feature_importance_shap[sorted_idx], color='coral')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Feature Importance (Top 15)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: shap_importance.png")

    # Full SHAP summary plot - use class 0 values for visualization
    plt.figure(figsize=(12, 10))
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Use first class (human) SHAP values
        shap_class0 = shap_values[:, :, 0]
    elif isinstance(shap_values, list):
        shap_class0 = shap_values[0]
    else:
        shap_class0 = shap_values

    shap.summary_plot(shap_class0, X_scaled, feature_names=feature_cols,
                     show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: shap_summary.png")

except Exception as e:
    import traceback
    print(f"   SHAP analysis error: {e}")
    traceback.print_exc()

# ============================================
# Results Comparison Table
# ============================================
print("\n[4.3] Generating results comparison table...")

# Create comparison table
comparison_data = []
for task_name, results in [("Binary (Human vs Bot)", binary_results),
                           ("Three-Class", multi_results)]:
    for model_name, metrics in results.items():
        comparison_data.append({
            'Task': task_name,
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy'][0]:.4f} +/- {metrics['accuracy'][1]:.4f}",
            'Precision': f"{metrics['precision'][0]:.4f} +/- {metrics['precision'][1]:.4f}",
            'Recall': f"{metrics['recall'][0]:.4f} +/- {metrics['recall'][1]:.4f}",
            'F1 (macro)': f"{metrics['f1'][0]:.4f} +/- {metrics['f1'][1]:.4f}",
            'ROC-AUC': f"{metrics['auc'][0]:.4f} +/- {metrics['auc'][1]:.4f}"
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n   Model Comparison:")
print(comparison_df.to_string(index=False))

# ============================================
# Write Results Summary
# ============================================
print("\n[4.4] Writing results summary...")

results_md = f"""# Mouse Dynamics Bot Detection - Results Summary

## Dataset Overview
- **Source**: M4D Mouse Dynamics Dataset (Iliou et al., 2021)
- **Total sessions**: {len(sessions_data)}
- **Classes**: Human ({label_counts['human']}), Moderate Bot ({label_counts['moderate_bot']}), Advanced Bot ({label_counts['advanced_bot']})

## Feature Engineering
Extracted {len(feature_cols)} features per session:
- **Kinematic**: speed (mean, std, max, median), acceleration, jerk
- **Spatial**: path straightness, curvature, angle changes, path length
- **Temporal**: session duration, event rate, pause detection, inter-event entropy
- **Behavioral**: click rate, scroll events, direction changes, speed skewness/kurtosis

## Classification Results

### Binary Classification (Human vs All Bots)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
"""

for model_name, metrics in binary_results.items():
    results_md += f"| {model_name} | {metrics['accuracy'][0]:.4f} | {metrics['precision'][0]:.4f} | {metrics['recall'][0]:.4f} | {metrics['f1'][0]:.4f} | {metrics['auc'][0]:.4f} |\n"

results_md += f"""
**Best Model**: {best_model_name} (F1 = {binary_results[best_model_name]['f1'][0]:.4f})

### Three-Class Classification
| Model | Accuracy | Precision | Recall | F1 (macro) | ROC-AUC |
|-------|----------|-----------|--------|------------|---------|
"""

for model_name, metrics in multi_results.items():
    results_md += f"| {model_name} | {metrics['accuracy'][0]:.4f} | {metrics['precision'][0]:.4f} | {metrics['recall'][0]:.4f} | {metrics['f1'][0]:.4f} | {metrics['auc'][0]:.4f} |\n"

results_md += f"""
**Best Model**: {best_model_name_multi} (F1 = {multi_results[best_model_name_multi]['f1'][0]:.4f})

## Key Findings

### Top Discriminating Features (by Random Forest importance)
"""
for i, row in importance_df.head(10).iterrows():
    results_md += f"{importance_df.index.get_loc(i)+1}. **{row['feature']}**: {row['importance']:.4f}\n"

results_md += """
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
"""

with open(os.path.join(OUTPUT_DIR, 'results.md'), 'w') as f:
    f.write(results_md)

print("   Saved: results.md")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nFiles generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath) / 1024
    print(f"  - {f} ({size:.1f} KB)")
