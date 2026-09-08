import pandas as pd
import numpy as np
from tqdm import tqdm
# Target Cohort stay_ids
class target_cohort:
    def __init__(self,withclause,fromwhereclause,conn,cur):
        self.withclause=withclause
        self.fromwhereclause=fromwhereclause
        self.conn=conn
        self.cur=cur
    def query(self):
        q=f"""
        {self.whereclause}
        Select subject_id, stay_id, intime {self.fromwhereclause};
        """
        self.cur.execute(q)
        q_stay_id_result=self.cur.fetchall()
        first_stay_id=[]
        for i in q_stay_id_result:
            first_stay_id.append(i['stay_id'])
        first_stay_id_str=", ".join(map(str, first_stay_id))
        return first_stay_id_str

# States Data Preprocessing
class states_preprocessor:
    def __init__(self,conn,cur,INTERVAL,my_required_items,first_stay_id,initial_values,zero_fill_cols):
        self.conn=conn
        self.cur=cur
        self.INTERVAL=INTERVAL
        self.initial_values=initial_values
        self.first_stay_id=first_stay_id
        self.my_required_items=my_required_items
        self.zero_fill_cols=zero_fill_cols

    def query(config, stay_id_list, conn): # fetch queries
        stay_str = ','.join(map(str, stay_id_list))
        sql = f"""
            WITH item_filtered AS (
                SELECT {config['stay_id_col']} AS stay_id, 
                    {config['time_col']} AS charttime, 
                    {config['value_col']} AS {config['item_name']}
                FROM {config['table_name']}
                {config['join_clause']}
                WHERE itemid = {config['item_id']}
            )
            SELECT stay_id, charttime, {config['item_name']}
            FROM item_filtered
            WHERE stay_id IN ({stay_str})
        """
        return pd.read_sql(sql, conn)
    
    def remove_outliers(config, data): # removing ~1,99~ quantile as an outliers
        item = config['item_name']
        if not data.empty and item in data.columns:
            q_low = data.groupby('stay_id')[item].transform(lambda x: x.quantile(0.01))
            q_hi = data.groupby('stay_id')[item].transform(lambda x: x.quantile(0.99))
            data = data[(data[item] >= q_low) & (data[item] <= q_hi)]
        return data
    
    def resampling(config, data): # resampling datas w.r.t given resampling hour
        data['charttime'] = pd.to_datetime(data['charttime'])
        resampled = (data.groupby('stay_id')
                    .apply(lambda x: x.set_index('charttime')
                                    .resample(f"{config['resampling_hour']}h")[config['item_name']]
                                    .agg(config['resample_method']))
                    .reset_index())
        return resampled
    
    def preprocessing(config, data, initial_values): # preprocessing
        item = config['item_name']
        method = config['fill_method']
        if initial_values and item in initial_values and not data.empty:
            first_indices = data.groupby('stay_id').head(1).index
            data.loc[first_indices, item] = data.loc[first_indices, item].fillna(initial_values[item])

        if method == 'ffill':
            data[item] = data.groupby('stay_id')[item].ffill()
        elif method == 'bfill':
            data[item] = data.groupby('stay_id')[item].bfill()
        elif method == 'interpolate':
            data[item] = data.groupby('stay_id')[item].transform(lambda x: x.interpolate())
            
        return data
    
    def create_pipeline_config(required_items,global_resample_hour=1): # creating preprocessing guidelines
        config_list = []
        for item in required_items:
            config = {
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'table_name': item.get('table_name', 'mimic.chartevents'),
                'time_col': item.get('time_col', 'charttime'),
                'value_col': item.get('value_col', 'valuenum'),
                'stay_id_col': item.get('stay_id_col', 'stay_id'),
                'join_clause': item.get('join_clause', ''),
                'resampling_hour': item.get('resampling_hour', global_resample_hour),
                'resample_method': item.get('resample_method', 'mean'),
                'fill_method': item.get('fill_method', 'interpolate')
            }
            config_list.append(config)
        return config_list

    def get_data(config_list, stay_id_list, conn, initial_values): # main pipe line
        extracted_data = {}
        
        for config in tqdm(config_list):
            raw_df = query(config, stay_id_list, conn)
            filtered_df = remove_outliers(config, raw_df)
            resampled_df = resampling(config, filtered_df)
            extracted_data[config['item_name']] = resampled_df

        final_df = None
        for name, df in extracted_data.items():
            if final_df is None:
                final_df = df
            else:
                final_df = pd.merge(final_df, df, on=['stay_id', 'charttime'], how='outer')

        if final_df is not None:
            final_df = final_df.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

        if 'BPM_inv' in final_df.columns and 'BPM_noninv' in final_df.columns:
            final_df['heart_rate'] = final_df['BPM_inv'].fillna(final_df['BPM_noninv'])
            final_df = final_df.drop(columns=['BPM_inv', 'BPM_noninv'])
        elif 'BPM_inv' in final_df.columns:
            final_df['heart_rate'] = final_df['BPM_inv']
        elif 'BPM_noninv' in final_df.columns:
            final_df['heart_rate'] = final_df['BPM_noninv']

        if 'Temperature' in final_df.columns and 'heart_rate' in final_df.columns and 'RR' in final_df.columns and 'WBC' in final_df.columns:
            final_df['SIRS'] = final_df.apply(calculate_sirs, axis=1)
        
        if 'heart_rate' in final_df.columns and 'NIBPs' in final_df.columns:
            final_df['shock_index'] = final_df.apply(shock_index, axis=1)
            
        final_df['vasopressor_eq'] = final_df.apply(cal_vasopressor, axis=1)

        for config in tqdm(config_list):
            item = config['item_name']
            if item in ['BPM_inv', 'BPM_noninv']:
                item = 'heart_rate'
                
            if item not in final_df.columns:
                continue
            final_df[item] = pd.to_numeric(final_df[item], errors='coerce')
            if initial_values and item in initial_values:
                first_indices = final_df.groupby('stay_id').head(1).index
                final_df.loc[first_indices, item] = final_df.loc[first_indices, item].fillna(initial_values[item])
                
            method = config['fill_method']
            if method == 'ffill':
                final_df[item] = final_df.groupby('stay_id')[item].ffill()
            elif method == 'bfill':
                final_df[item] = final_df.groupby('stay_id')[item].bfill()
            elif method == 'interpolate':
                final_df[item] = final_df.groupby('stay_id')[item].transform(lambda x: x.interpolate().ffill().bfill())
                
        for computed_col in ['SIRS', 'shock_index', 'vasopressor_eq']:
            if computed_col in final_df.columns:
                final_df[computed_col] = final_df.groupby('stay_id')[computed_col].ffill().fillna(0)

        return final_df

    # calculation functions
    def calculate_sirs(row): # SIRS 계산
        count = 0
        if pd.notna(row.get('Temperature')) and (row['Temperature'] > 38 or row['Temperature'] < 36):
            count += 1
        if pd.notna(row.get('heart_rate')) and row['heart_rate'] > 90:
            count += 1
        if pd.notna(row.get('RR')) and row['RR'] > 20:
            count += 1
        if pd.notna(row.get('WBC')) and (row['WBC'] > 12000 or row['WBC'] < 4000):
        count += 1  
        return count if count >= 2 else 0

    def shock_index(row): # shock index 계산
        hr = row.get('heart_rate')
        sbp = row.get('NIBPs')
        if pd.isna(hr) or pd.isna(sbp) or sbp == 0:
            return np.nan
        return hr / sbp

    def merge_bpm(df_inv, df_noninv): # bpm 전처리
        merged = pd.merge(df_inv, df_noninv, on=['stay_id', 'charttime'], how='outer')
        merged['BPM'] = merged['BPM_inv'].fillna(merged['BPM_noninv'])
        merged = merged.drop(columns=['BPM_inv', 'BPM_noninv'])
        return merged

    def fill_zero(df, zero_fill_cols): # 0 보간
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def get_ages(): # stay_id별로 나이 계산
        sql_age=f"""
            With group_age as
            (select i.stay_id,
            (p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year)) AS age 
            FROM mimic.icustays i
            LEFT JOIN mimic_hosp.admissions a ON i.hadm_id = a.hadm_id
            LEFT JOIN mimic.patients p ON i.subject_id = p.subject_id)
            select group_age.stay_id,group_age.age
            from group_age where 
            group_age.stay_id in ({first_stay_id_str})
            order by group_age.age desc;"""
        cur.execute(sql_age)
        age_result=cur.fetchall()
        df_age=pd.DataFrame(age_result)
        return df_age

    def main(self):
        config_list = self.create_pipeline_config(self.my_required_items,global_resample_hour=self.INTERVAL)
        final_dataframe = self.get_data(config_list, self.first_stay_id, self.conn, self.initial_values)
        final_dataframe = self.fill_zero(final_dataframe, self.zero_fill_cols)
        return final_dataframe

# Action Data Preprocessing
class action_preprocessor:
    def __init__(self,conn,cur,resample_hour,stay_id_list):
        self.conn=conn
        self.cur=cur
        self.stay_id_list=stay_id_list
        self.resample_hour=resample_hour

    def get_action_data(self): # preprocessing action data 
        stay_str = ','.join(map(str, self.stay_id_list))
        
        query = f"""
            SELECT * 
            FROM value_based_data.treatment_features 
            WHERE stay_id IN ({stay_str})
            ORDER BY stay_id, time_hour
        """
        action_df = pd.read_sql(query, con=self.conn)
        
        if action_df.empty:
            print("조회된 데이터가 없습니다.")
            return pd.DataFrame()

        action_df['time_hour'] = pd.to_datetime(action_df['time_hour'], errors='coerce')
        
        vasopressors = ["norepinephrine", 'dopamine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dobutamine']
        vaso_amounts = [v + "_amount" for v in vasopressors]
        excluded_base = ['angiotensin_ii', 'Phenylephrine50250', 'Phenylephrine200250_old', 'Phenylephrine200250']
        excluded = excluded_base + [e + "_amount" for e in excluded_base]
        TIME_COL = "time_hour"
        
        non_iv_cols = vasopressors + excluded + ['time_bin', TIME_COL, 'stay_id', 'event_id', 'vaso', 'patient_weight'] + vaso_amounts
        
        processed_dfs = []
        grouped_data = action_df.groupby('stay_id')
        
        for stay_id, df in tqdm(grouped_data, desc=f"Processing Actions ({self.resample_hour}h interval)"):
            df = df.reset_index(drop=True)
            
            min_time = df[TIME_COL].min()
            max_time = df[TIME_COL].max()
            if pd.isna(min_time) or pd.isna(max_time):
                continue
            time_grid = pd.date_range(
                start=min_time.floor('h'), 
                end=max_time.ceil('h') + pd.Timedelta(hours=self.resample_hour), 
                freq=f'{self.resample_hour}h'
            )
            
            df.columns = df.columns.astype(str).str.strip()
            cols_to_keep = [col for col in df.columns if col not in excluded]
            df_filtered = df[cols_to_keep].copy()
            
            df_filtered['time_bin'] = pd.cut(df_filtered[TIME_COL], bins=time_grid, right=True)
            
            agg_dict = {}
            for col in df_filtered.columns:
                if col in [TIME_COL, 'time_bin', 'stay_id', 'event_id']:
                    continue
                elif col == 'patient_weight':
                    agg_dict[col] = 'mean'
                elif col in vasopressors:
                    agg_dict[col] = 'max'
                else:
                    agg_dict[col] = 'sum'
                    
            df_grouped = df_filtered.groupby('time_bin', observed=False).agg(agg_dict).fillna(0)
            df_grouped = df_grouped.reset_index()
            df_grouped['stay_id'] = stay_id
            
            df_grouped['vaso'] = df_grouped.apply(self.cal_vasopressor_action, axis=1)
            
            iv_cols = [col for col in df_grouped.columns if col not in non_iv_cols]
            df_grouped['iv_fluid'] = df_grouped[iv_cols].sum(axis=1)
            
            df_grouped['iv_action'] = df_grouped['iv_fluid'].apply(self.discretize_iv_fluid)
            df_grouped['vaso_action'] = df_grouped['vaso'].apply(self.discretize_vasopressor)
            df_grouped['final_action'] = (df_grouped['iv_action'] - 1) * 5 + (df_grouped['vaso_action'] - 1)
            
            processed_dfs.append(df_grouped)
            
        if processed_dfs:
            return pd.concat(processed_dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    #calculation functions 
    def discretize_iv_fluid(val):# discretizing IV into 5 discrete actions
        if val == 0: return 1
        elif 0 < val <= 50: return 2
        elif 50 < val <= 180: return 3
        elif 180 < val <= 530: return 4
        else: return 5

    def discretize_vasopressor(val): # discretizing Vaso into 5 discrete actions
        if val == 0: return 1
        elif 0 < val <= 0.08: return 2
        elif 0.08 < val <= 0.22: return 3
        elif 0.22 < val <= 0.45: return 4
        else: return 5

    def cal_vasopressor_action(row): # vaso transformation
        norepi = row.get('norepinephrine', 0)
        dopa = row.get('dopamine', 0)
        epi = row.get('epinephrine', 0)
        phenyl = row.get('phenylephrine', 0)
        vaso = row.get('vasopressin', 0)
        return norepi + (1/150)*dopa + 0.1*epi + 0.1*phenyl + (2.5*vaso)/60

    def main(self):
        final_action_dataframe = self.get_action_data()
        return final_action_dataframe


