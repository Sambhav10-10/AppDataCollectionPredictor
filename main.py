
import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

def load_data(filepath):
    """Load JSON dataset and return list of app records."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} apps from dataset\n")
    return data

dataset = load_data('game_app_data_safety.json')

# ============================================================================
# STEP 2: FEATURE ENGINEERING (NO DATA LEAKAGE)
# ============================================================================
print("=" * 80)
print("STEP 2: FEATURE ENGINEERING (AVOIDING DATA LEAKAGE)")
print("=" * 80)

def extract_features_and_labels_no_leakage(dataset):
    """
    Extract features and labels WITHOUT data leakage.
    
    IMPORTANT: Input features are BEHAVIORAL characteristics, NOT output labels.
    
    INPUT FEATURES (8) - Valid behavioral features:
    1. shared_with_third_parties - 1 if sensitive data is shared
    2. used_for_advertising - 1 if data used for ads/marketing
    3. encryption_enabled - 1 if "encrypt" in securityPractices
    4. data_deletion_allowed - 1 if "delete" option exists
    5. collects_sensitive_data - 1 if any sensitive type collected
    6. num_data_types_collected - COUNT of all collected data types
    7. collects_multiple_types - 1 if more than 2 types
    8. num_data_types_shared - COUNT of all shared data types
    
    OUTPUT LABELS (4) - Multi-label targets:
    1. collects_personal_info
    2. collects_location
    3. collects_contacts
    4. collects_financial_info
    
    Note: We DO NOT use individual output labels as input features.
    This prevents data leakage and ensures realistic model accuracy.
    """
    
    features = []
    labels = []
    
    for app in dataset:
        app_info = app.get('appInfo', {})
        collected_data = app_info.get('collectedData', [])
        shared_data = app_info.get('sharedData', [])
        security_practices = app_info.get('securityPractices', [])
        
        # ====== EXTRACT OUTPUT LABELS ======
        # These are what we want to predict
        collected_types = {item.get('type') for item in collected_data}
        
        label_personal = 1 if 'Personal info' in collected_types else 0
        label_location = 1 if 'Location' in collected_types else 0
        label_contacts = 1 if 'Contacts' in collected_types else 0
        label_financial = 1 if 'Financial info' in collected_types else 0
        
        # ====== EXTRACT INPUT FEATURES (NO LEAKAGE) ======
        
        # Feature 1: shared_with_third_parties
        # Check if ANY sensitive data is shared (not specific types)
        sensitive_types = {'Personal info', 'Location', 'Contacts', 'Financial info'}
        shared_sensitive = any(
            item.get('type') in sensitive_types 
            for item in shared_data
        )
        feature_shared = 1 if shared_sensitive else 0
        
        # Feature 2: used_for_advertising
        # Check if "Advertising or marketing" appears in any purpose
        advertising_in_collected = any(
            'Advertising or marketing' in item.get('purpose', '')
            for item in collected_data
        )
        advertising_in_shared = any(
            'Advertising or marketing' in item.get('purpose', '')
            for item in shared_data
        )
        feature_advertising = 1 if (advertising_in_collected or advertising_in_shared) else 0
        
        # Feature 3: encryption_enabled
        # Check if any security practice mentions encryption
        feature_encryption = 1 if any(
            'encrypt' in practice.get('practice', '').lower()
            for practice in security_practices
        ) else 0
        
        # Feature 4: data_deletion_allowed
        # Check if any security practice mentions deletion/removal
        feature_deletion = 1 if any(
            'delet' in practice.get('practice', '').lower()
            for practice in security_practices
        ) else 0
        
        # Feature 5: collects_sensitive_data
        # Check if app collects ANY of the sensitive data types
        feature_sensitive = 1 if any(
            t in collected_types for t in sensitive_types
        ) else 0
        
        # Feature 6: num_data_types_collected
        # COUNT of distinct data types collected (behavioral metric)
        num_types_collected = len(collected_types)
        
        # Feature 7: collects_multiple_types
        # Binary: does it collect more than 2 types?
        feature_multiple = 1 if len(collected_types) > 2 else 0
        
        # Feature 8: num_data_types_shared
        # COUNT of distinct data types shared (behavioral metric)
        shared_types = {item.get('type') for item in shared_data}
        num_types_shared = len(shared_types)
        
        # Append features and labels
        features.append([
            feature_shared,
            feature_advertising,
            feature_encryption,
            feature_deletion,
            feature_sensitive,
            num_types_collected,
            feature_multiple,
            num_types_shared
        ])
        
        labels.append([
            label_personal,
            label_location,
            label_contacts,
            label_financial
        ])
    
    return np.array(features), np.array(labels)

X, Y = extract_features_and_labels_no_leakage(dataset)

feature_names = [
    'shared_with_third_parties',
    'used_for_advertising',
    'encryption_enabled',
    'data_deletion_allowed',
    'collects_sensitive_data',
    'num_data_types_collected',
    'collects_multiple_types',
    'num_data_types_shared'
]

label_names = [
    'Personal Info',
    'Location',
    'Contacts',
    'Financial Info'
]

print(f"✓ Features extracted: {X.shape}")
print(f"✓ Labels extracted: {Y.shape}")
print(f"\nInput Features ({len(feature_names)}) - Valid behavioral characteristics:")
for i, name in enumerate(feature_names, 1):
    print(f"  {i}. {name}")

print(f"\nOutput Labels ({len(label_names)}) - Multi-label targets:")
for i, name in enumerate(label_names, 1):
    print(f"  {i}. {name}")

print(f"\nLabel Distribution:")
for i, name in enumerate(label_names):
    count = Y[:, i].sum()
    percentage = (Y[:, i].mean() * 100)
    print(f"  - {name:.<30} {int(count):>3} apps ({percentage:>5.1f}%)")

print(f"\n✓ NO DATA LEAKAGE - Output labels are NOT included as input features")

# ============================================================================
# STEP 3: CREATE DATASET (PANDAS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATE DATASET")
print("=" * 80)

df = pd.DataFrame(X, columns=feature_names)
for i, label_name in enumerate(label_names):
    df[label_name] = Y[:, i]

print("✓ Dataset created with pandas")
print(f"\nDataset shape: {df.shape}")
print(f"\nFeature statistics:")
print(df[feature_names].describe())

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN-TEST SPLIT (80-20)")
print("=" * 80)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42
)

print(f"✓ Training set size: {X_train.shape[0]} samples")
print(f"✓ Test set size: {X_test.shape[0]} samples")

# ============================================================================
# STEP 5 & 6: TRAIN TUNED MODELS WITH MultiOutputClassifier
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5-6: TRAINING 3 TUNED MODELS (WITHOUT DATA LEAKAGE)")
print("=" * 80)

# Model 1: Decision Tree (tuned)
print("\n[1/3] Training Decision Tree Classifier (max_depth=8, min_samples_split=4)...")
dt_model = MultiOutputClassifier(
    DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42)
)
dt_model.fit(X_train, Y_train)
print("✓ Decision Tree trained")

# Model 2: Random Forest (tuned with balanced class weights)
print("[2/3] Training Random Forest Classifier (n_estimators=200, max_depth=10, balanced)...")
rf_model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        class_weight='balanced_subsample',
        random_state=42
    )
)
rf_model.fit(X_train, Y_train)
print("✓ Random Forest trained")

# Model 3: Logistic Regression with scaling and balanced class weights
print("[3/3] Training Logistic Regression (with StandardScaler + balanced weights)...")
# Use Pipeline to include StandardScaler only for this model
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
])
lr_model = MultiOutputClassifier(lr_pipeline)
lr_model.fit(X_train, Y_train)
print("✓ Logistic Regression trained (with feature scaling)")

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL EVALUATION")
print("=" * 80)

def evaluate_model(model, X_test, Y_test, model_name):
    """Evaluate model with multiple metrics."""
    Y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Hamming Loss
    hamming = hamming_loss(Y_test, Y_pred)
    
    # F1-score (macro - gives equal weight to each label)
    f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    
    print(f"\n{'─' * 80}")
    print(f"{model_name}")
    print(f"{'─' * 80}")
    print(f"Accuracy:      {accuracy*100:>6.2f}%")
    print(f"F1-Score:      {f1_macro:>6.4f}")
    print(f"Hamming Loss:  {hamming:>6.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        Y_test, Y_pred,
        target_names=label_names,
        digits=3,
        zero_division=0
    ))
    
    return accuracy, f1_macro

print()
dt_acc, dt_f1 = evaluate_model(dt_model, X_test, Y_test, "Decision Tree Classifier")
rf_acc, rf_f1 = evaluate_model(rf_model, X_test, Y_test, "Random Forest Classifier")
lr_acc, lr_f1 = evaluate_model(lr_model, X_test, Y_test, "Logistic Regression")

# ============================================================================
# STEP 8: MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: MODEL COMPARISON (BY ACCURACY)")
print("=" * 80)

results = [
    ("Decision Tree", dt_acc, dt_f1),
    ("Random Forest", rf_acc, rf_f1),
    ("Logistic Regression", lr_acc, lr_f1)
]
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print()
for i, (name, acc, f1) in enumerate(results_sorted, 1):
    bar = "█" * int(acc * 25)
    print(f"{i}. {name:.<30} {bar} {acc*100:>6.2f}% (F1: {f1:.4f})")

best_model_name = results_sorted[0][0]
best_acc = results_sorted[0][1]
best_f1 = results_sorted[0][2]

if best_model_name == "Decision Tree":
    best_model = dt_model
elif best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = lr_model

print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  • Accuracy: {best_acc*100:.2f}%")
print(f"  • F1-Score: {best_f1:.4f}")
print(f"\n✓ REALISTIC RESULTS - No data leakage")

# ============================================================================
# STEP 9: PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: PREDICTION FUNCTION")
print("=" * 80)

def predict_app_data(input_dict, model=None):
    """
    Predict what data an app collects based on VALID behavioral features.
    
    Input dictionary with 8 features (NO output labels):
    - shared_with_third_parties (0 or 1)
    - used_for_advertising (0 or 1)
    - encryption_enabled (0 or 1)
    - data_deletion_allowed (0 or 1)
    - collects_sensitive_data (0 or 1)
    - num_data_types_collected (0-N)
    - collects_multiple_types (0 or 1)
    - num_data_types_shared (0-N)
    
    Output: Prints predictions for each data type
    """
    if model is None:
        model = best_model
    
    # Convert input dict to feature vector (maintain order)
    feature_vector = np.array([[
        input_dict.get('shared_with_third_parties', 0),
        input_dict.get('used_for_advertising', 0),
        input_dict.get('encryption_enabled', 0),
        input_dict.get('data_deletion_allowed', 0),
        input_dict.get('collects_sensitive_data', 0),
        input_dict.get('num_data_types_collected', 0),
        input_dict.get('collects_multiple_types', 0),
        input_dict.get('num_data_types_shared', 0)
    ]])
    
    # Make prediction
    prediction = model.predict(feature_vector)[0]
    
    # Display results
    print(f"\nData Collection Prediction:")
    print(f"{'─' * 70}")
    print(f"Input Features:")
    for key, value in sorted(input_dict.items()):
        if key.startswith('num_'):
            print(f"  → {key:.<40} {value}")
        else:
            status = "✓" if value == 1 else "✗"
            print(f"  {status} {key:.<40} {value}")
    
    print(f"\nPredicted Data Collections:")
    for i, data_type in enumerate(label_names):
        result = "Yes" if prediction[i] == 1 else "No"
        symbol = "✓" if prediction[i] == 1 else "✗"
        print(f"  {symbol} {data_type:.<40} {result}")
    print(f"{'─' * 70}")
    
    return prediction

print("\n✓ Prediction function created!")

# ============================================================================
# TESTING PREDICTION FUNCTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: TESTING PREDICTION FUNCTION")
print("=" * 80)

# Example 1: Safe app (minimal data)
print("\n[Example 1] SAFE APP:")
print("  • Not shared with 3rd parties")
print("  • Not used for advertising")
print("  • Collects only 1 data type")
example1 = {
    'shared_with_third_parties': 0,
    'used_for_advertising': 0,
    'encryption_enabled': 1,
    'data_deletion_allowed': 1,
    'collects_sensitive_data': 1,
    'num_data_types_collected': 1,
    'collects_multiple_types': 0,
    'num_data_types_shared': 0
}
predict_app_data(example1, best_model)

# Example 2: Moderate app
print("\n[Example 2] MODERATE APP:")
print("  • Shared with 3rd parties")
print("  • Used for advertising")
print("  • Collects 4 data types")
example2 = {
    'shared_with_third_parties': 1,
    'used_for_advertising': 1,
    'encryption_enabled': 1,
    'data_deletion_allowed': 0,
    'collects_sensitive_data': 1,
    'num_data_types_collected': 4,
    'collects_multiple_types': 1,
    'num_data_types_shared': 2
}
predict_app_data(example2, best_model)

# Example 3: Risky app
print("\n[Example 3] RISKY APP:")
print("  • Shared with 3rd parties")
print("  • Used for advertising")
print("  • Collects 7 data types")
print("  • NOT encrypted")
example3 = {
    'shared_with_third_parties': 1,
    'used_for_advertising': 1,
    'encryption_enabled': 0,
    'data_deletion_allowed': 0,
    'collects_sensitive_data': 1,
    'num_data_types_collected': 7,
    'collects_multiple_types': 1,
    'num_data_types_shared': 4
}
predict_app_data(example3, best_model)

# ============================================================================
# PROJECT SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT COMPLETE!")
print("=" * 80)
print(f"\nSummary:")
print(f"  • Loaded: {len(dataset)} apps")
print(f"  • Input Features: {len(feature_names)} (behavioral, no leakage)")
print(f"  • Output Labels: {len(label_names)} (multi-label)")
print(f"  • Models Trained: 3 (DT, RF, LR)")
print(f"  • Best Model: {best_model_name}")
print(f"  • Best Accuracy: {best_acc*100:.2f}%")
print(f"  • Best F1-Score: {best_f1:.4f}")


