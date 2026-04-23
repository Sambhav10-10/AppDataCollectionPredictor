import json
import pandas as pd

with open('game_app_data_safety.json') as f:
    data = json.load(f)

def extract_features(apps_data):
    features_list = []
    for app in apps_data:
        app_info = app.get('appInfo', {})
        collected_data = app_info.get('collectedData', [])
        shared_data = app_info.get('sharedData', [])
        security_practices = app_info.get('securityPractices', [])
        
        features = {}
        sensitive_types = {'Contacts', 'Personal info', 'Location', 'Financial info', 'Photos and videos'}
        collected_sensitive = [d for d in collected_data if d.get('type') in sensitive_types]
        features['collects_sensitive_data'] = 1 if collected_sensitive else 0
        mandatory_sensitive = [d for d in collected_sensitive if not d.get('optional', True)]
        features['collects_mandatory_sensitive'] = 1 if mandatory_sensitive else 0
        shared_sensitive = [d for d in shared_data if d.get('type') in sensitive_types]
        features['shared_with_third_parties'] = 1 if shared_sensitive else 0
        
        advertising_purpose = 'Advertising or marketing'
        advertising_collected = any(advertising_purpose in d.get('purpose', '') for d in collected_data)
        advertising_shared = any(advertising_purpose in d.get('purpose', '') for d in shared_data)
        features['used_for_advertising'] = 1 if (advertising_collected or advertising_shared) else 0
        
        encryption_practice = any('encrypt' in p.get('practice', '').lower() for p in security_practices)
        features['encryption_enabled'] = 1 if encryption_practice else 0
        deletion_practice = any('delete' in p.get('practice', '').lower() for p in security_practices)
        features['data_deletion_allowed'] = 1 if deletion_practice else 0
        
        features_list.append(features)
    return pd.DataFrame(features_list)

df = extract_features(data)

# Check mandatory+shared with both protections
mandatory_shared = df[(df['collects_mandatory_sensitive'] == 1) & (df['shared_with_third_parties'] == 1)]
both_protected = mandatory_shared[(mandatory_shared['encryption_enabled'] == 1) & (mandatory_shared['data_deletion_allowed'] == 1)]
print('Mandatory+shared apps:', len(mandatory_shared))
print('  With BOTH encryption+deletion:', len(both_protected))
print('  With encryption only (no deletion):', (mandatory_shared[(mandatory_shared['encryption_enabled'] == 1) & (mandatory_shared['data_deletion_allowed'] == 0)]).shape[0])

# Check sensitive+ads+deletion
sensitive_ads = df[(df['collects_sensitive_data'] == 1) & (df['used_for_advertising'] == 1)]
print('\nSensitive+ads apps:', len(sensitive_ads))
print('  With deletion:', (sensitive_ads['data_deletion_allowed'] == 1).sum())
print('  Without deletion:', (sensitive_ads['data_deletion_allowed'] == 0).sum())

# Check sensitive+shared+ads with encryption
sensitive_shared_ads = df[(df['collects_sensitive_data'] == 1) & (df['shared_with_third_parties'] == 1) & (df['used_for_advertising'] == 1)]
print('\nSensitive+shared+ads apps:', len(sensitive_shared_ads))
print('  With encryption:', (sensitive_shared_ads['encryption_enabled'] == 1).sum())
print('  Without encryption:', (sensitive_shared_ads['encryption_enabled'] == 0).sum())
