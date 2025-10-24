# centralized/preprocess_data.py

import os
import time
import argparse
import numpy as np
import pandas as pd

# ===================================================================
# 0. Column Configurations
# ===================================================================

cfg = {
    'fraud_yn':           'FRAUD_YN(이상거래 여부)',
    'amount':             'log_amt(거래금액)',
    'date':               'log_real_date(본거래일)',
    'time':               'log_real_time(본거래시간)',
    'channel':            'log_ch_cd(채널코드)',
    'ip_country':         'cntry_cd(공인IP 국가코드)',
    'proxy_yn':           'prxy_yn(프록시 설정 여부)',
    'proxy_country':      'prxy_cntry_cd(프록시 IP 국가코드)',
    'remote_yn':          'WIN_OS_RMT_YN(윈도우 운영체제 원격접속 설정 여부)',
    'remote_type':        'TYPE_RMT(원격접속 프로그램 종류)',
    'remote_country':     'WIN_OS_IP_RMT_CNTRY(윈도우 운영체제 원격접속 IP 국가코드)',
    'malware_yn':         'MALWARE_DETECT_YN(악성앱 탐지여부)',
    'tamper_yn':          'APP_TAMPERING_YN(앱위변조 탐지여부)',
    'rooting_yn':         'ROOTING_YN(루팅 탐지여부)',
    'yfds_version':       'YFDS_VERSION(PC수집기 버전)',
    'remote_rename_map': {'1.0': 'Win', '2.0': 'Chrm', '3.0': 'Tvw'},
}


def parse_arguments():
    parser = argparse.ArgumentParser('Preprocess raw CSV data for modeling (Centralized Setting).')
    parser.add_argument('--input_dir', default='../dataset', help='Directory containing the raw input CSV file')
    parser.add_argument('--save_dir',  default='../dataset', help='Directory to save the processed CSV file')
    return parser.parse_args()


def preprocess_data(raw_df):
    new_df = raw_df.copy()
    original_columns = set(new_df.columns)

    # Label
    new_df = string_to_binary(new_df, cfg['fraud_yn'], 'label')  # 0: normal; 1: fraud

    # F1: Amount_bin
    new_df = categorize_amt(new_df, cfg['amount'])

    # F2~F3: Weekday and Weekend
    new_df = add_weekday_weekend(new_df, cfg['date'])

    # F4: Hour of day
    new_df = time_to_hour(new_df, cfg['time'])

    # F5: Is channel web (1:Web, 0:Mobile)
    new_df = channel_code_to_boolean(new_df, cfg['channel'])

    # F6: Transaction country
    new_df = add_onehot_features(new_df, cfg['ip_country'], 'cc')

    # F7: Is transaction domestic
    new_df['is_domestic'] = new_df[cfg['ip_country']].astype('string').str.upper().str.strip().eq('KR').astype('Int8')

    # F8: Is proxy used
    new_df = string_to_binary(new_df, cfg['proxy_yn'], 'is_prxy_used')

    # F9: Is proxy domestic
    new_df = add_onehot_features(new_df, cfg['proxy_country'], 'ipd', domestic='KR')

    # F10: Proxy country
    new_df = add_onehot_features(new_df, cfg['proxy_country'], 'pc')

    # F11: Is rmt used
    new_df = add_onehot_features(new_df, cfg['remote_yn'], 'rm')

    # F12: What types of rmt is used
    new_df = add_onehot_features(new_df, cfg['remote_type'], 'tr', rename_map=cfg['remote_rename_map'])

    # F13: Rmt country
    new_df = add_onehot_features(new_df, cfg['remote_country'], 'rc')

    # F14: Is rmt domestic
    new_df = add_onehot_features(new_df, cfg['remote_country'], 'ird', domestic='KR')

    # F15~17: is malware,app tampering, rooting used
    new_df = string_to_binary(new_df, cfg['malware_yn'], 'is_malware_used')
    new_df = string_to_binary(new_df, cfg['tamper_yn'], 'is_app_tampering_used')
    new_df = string_to_binary(new_df, cfg['rooting_yn'], 'is_rooting_used')

    # F18: VFDS version
    new_df = add_onehot_features(new_df, cfg['yfds_version'], 'yv')

    new_columns = set(new_df.columns) - original_columns
    final_feature_columns = sorted(list(new_columns))

    # Re-arrange so that the label column is the last column
    final_feature_columns.remove('label')
    final_feature_columns.append('label')

    return new_df[final_feature_columns]


# ===================================================================
# 1. General feature extraction helpers
# ===================================================================

def string_to_binary(passed_df, src_col_name, new_col_name):
    """ Convert 'Y'/'N' strings to binary 1/0 integers. """
    mapping_dict = {'Y': 1, 'N': 0}
    # Create a copy and apply mapping
    new_df = passed_df.copy()
    series = passed_df[src_col_name].map(mapping_dict)
    new_df[new_col_name] = series.astype('Int8')
    return new_df


def add_onehot_features(passed_df, src_col_name, prefix, domestic = None, rename_map=None):
    new_df = passed_df.copy()

    s = new_df[src_col_name].astype('string').str.upper().str.strip()
    s = s.fillna('Unknown')

    if domestic is not None:
        conditions = [
            s.eq(str(domestic).upper()),
            s.eq('Unknown')
        ]
        choices = [
            'Y',
            'Unknown'
        ]
        mapped_values = np.select(conditions, choices, default='N')
        series = pd.Series(mapped_values, index=new_df.index, name=src_col_name)
    else:
        series = s

    CATS = sorted(series.dropna().unique())
    s_categorical = pd.Categorical(series, categories=CATS)
    oh = pd.get_dummies(s_categorical, prefix=prefix, dtype='Int8')

    if rename_map is not None:
        full_rename_map = {f'{prefix}_{k}': f'{prefix}_{v}' for k, v in rename_map.items()}
        oh.rename(columns=full_rename_map, inplace=True)
        cols = [f'{prefix}_{rename_map.get(c, c)}' for c in CATS]
    else:
        cols = [f'{prefix}_{c}' for c in CATS]

    oh = oh.reindex(columns=cols, fill_value=0)

    # Merge the one-hot encoded columns back into the new DataFrame
    new_df = pd.concat([new_df, oh], axis=1)
    return new_df


# ===================================================================
# 2. More specific feature extraction
# ===================================================================

# Transaction amount
def categorize_amt(passed_df, src_col_name, new_col_name='amount_bin'):
    new_df = passed_df.copy()
    # domain bins in KRW: [0,1M), [1M,5M), [5M,10M), [10M, +inf)
    edges = [0, 1_000_000, 5_000_000, 10_000_000, np.inf]
    labels = [0, 1, 2, 3]  # 3 = >= 10M
    bins = pd.cut(
        new_df[src_col_name],
        bins=edges,
        labels=labels,
        right=False,            # [low, high)
        include_lowest=True,
        ordered=True
    )
    new_df[new_col_name] = bins.astype('Int64')  # keeps NA if any
    return new_df


# Weekday and Weekend
def add_weekday_weekend(passed_df, src_col_name ='date', weekday_col = 'weekday', weekend_col = 'is_weekend'):
    new_df = passed_df.copy()
    # parse YYYYMMDD; if already datetime, this is safe
    s = pd.to_datetime(new_df[src_col_name], format='%Y%m%d', errors='coerce')

    wk = s.dt.weekday                         # int 0..6, NaT -> NaN
    new_df[weekday_col]  = wk.astype('Int64') # nullable integer
    new_df[weekend_col]  = wk.ge(5).astype('Int8')  # True for Sat/Sun, NaT -> <NA>
    return new_df


# Hour of Day
def time_to_hour(passed_df, src_col_name, new_col_name='hour_of_day'):
    new_df = passed_df.copy()
    s = pd.Series(new_df[src_col_name]).astype('string').str.extract(r'(\d+)', expand=False)

    L = s.str.len()
    no_ms = L.le(6)
    with_ms = L.gt(6)

    s6 = s.where(no_ms).str.zfill(6)
    h6 = s6.str[0:2].astype('Int64')
    m6 = s6.str[2:4].astype('Int64')
    s6s = s6.str[4:6].astype('Int64')

    s9 = s.where(with_ms).str.zfill(9)
    h9 = s9.str[0:2].astype('Int64')
    m9 = s9.str[2:4].astype('Int64')
    s9s = s9.str[4:6].astype('Int64')
    ms9 = s9.str[6:9].astype('Int64')

    H = h6.fillna(h9)
    M = m6.fillna(m9)
    S = s6s.fillna(s9s)
    MS = ms9.fillna(0)

    # sanity checks; invalid → NaN
    valid = (H.between(0,23)) & (M.between(0,59)) & (S.between(0,59)) & (MS.between(0,999))
    hours = (H + M/60 + (S + MS/1000)/3600).where(valid, np.nan)

    new_df[new_col_name] = hours.astype('Float32')
    new_df[new_col_name] = new_df[new_col_name].astype('Int64')
    return new_df


# Transaction channel
def channel_code_to_boolean(passed_df, src_col_name, new_col_name='is_channel_web'):
    new_df = passed_df.copy()
    s = pd.to_numeric(new_df[src_col_name], errors='coerce').map({1: True, 2: False})
    new_df[new_col_name] = s.astype('Int8')
    return new_df


# ================================================================= #
# ========================= Main Function ========================= #
# ================================================================= #
def main():
    args = parse_arguments()
    input_dir, save_dir = args.input_dir, args.save_dir

    # Load raw data
    columns_to_load = [4, 6, 8, 10, 15, 16, 19, 29, 32, 63, 65, 67, 74, 77, 78, 79, 80]
    full_input_path = f'{input_dir}/raw.csv'
    raw_df = pd.read_csv(full_input_path, encoding='cp949', usecols=columns_to_load)

    # Preprocess data
    tic = time.time()
    processed_df = preprocess_data(raw_df)
    toc = time.time()
    print(f"Data preprocessing completed in {toc - tic:.2f} seconds.")

    # Filter out rows with missing hour_of_day
    processed_df_complete = processed_df[processed_df['hour_of_day'].notnull()].copy()
    print(f'Original data rows: {len(processed_df)}, after removing missing hour_of_day: {len(processed_df_complete)}')

    # Save processed data
    os.makedirs(save_dir, exist_ok=True)
    processed_data_path = os.path.join(save_dir, 'processed_data.csv')
    processed_data_no_missing_path = os.path.join(save_dir, 'processed_data_no_missing.csv')

    processed_df.to_csv(processed_data_path, index=False)
    processed_df_complete.to_csv(processed_data_no_missing_path, index=False)



if __name__ == '__main__':
    main()