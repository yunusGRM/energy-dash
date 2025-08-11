import azure.functions as func
import json
import logging
from datetime import datetime, timedelta, date
import pandas as pd
import requests
import pytz
import re
import numpy as np
import io
import time
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Energy dashboard function processed a request.')
    try:
        all_data = get_all_energy_data()
        fmt = req.params.get('format', 'html')
        if fmt == 'json':
            return func.HttpResponse(
                json.dumps(all_data, indent=2, default=str),
                mimetype="application/json",
                status_code=200
            )
        else:
            html_content = create_comprehensive_dashboard(all_data)
            return func.HttpResponse(
                html_content,
                mimetype="text/html",
                status_code=200
            )
    except Exception as e:
        logging.exception("Dashboard error")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)

def get_all_energy_data():
    data = {}
    try:
        data['interconnector'] = get_interconnector_data()
    except Exception as e:
        data['interconnector'] = f"Error: {str(e)}"
    try:
        data['nl_merit_order'] = get_nl_merit_order()
    except Exception as e:
        data['nl_merit_order'] = f"Error: {str(e)}"
    try:
        data['nl_balance_delta'] = get_nl_balance_delta()
    except Exception as e:
        data['nl_balance_delta'] = f"Error: {str(e)}"
    try:
        data['at_merit_order'] = get_at_merit_order()
    except Exception as e:
        data['at_merit_order'] = f"Error: {str(e)}"
    try:
        data['at_cross_zonal'] = get_at_cross_zonal_caps()
    except Exception as e:
        data['at_cross_zonal'] = f"Error: {str(e)}"
    data['timestamp'] = datetime.now().isoformat()
    return data

# ===== UK INTERCONNECTOR DATA =====
class AllInOneInterconnectorFetcher:
    def __init__(self):
        self.britned_api_base = "https://api.empire.britned.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Interconnector-API-Client/1.0'
        })
        self.rnp_interconnectors = {
            "IFA": ("FRGB", "GBFR"),
            "IFA2": ("FRGB", "GBFR"),
            "Nemo Link": ("BEGB", "GBBE"),
            "NSL": ("NOGB", "GBNO"),
            "Viking Link": ("D1GB", "GBD1"),
        }

    def extract_hour_from_time(self, time_string):
        try:
            match = re.search(r'\s(\d{2}):(\d{2})-', time_string)
            if match:
                hour = int(match.group(1))
                return f"H{hour + 1}"
            else:
                parts = time_string.split()
                if len(parts) >= 2:
                    time_part = parts[1]
                    hour_match = re.match(r'(\d{1,2}):', time_part)
                    if hour_match:
                        hour = int(hour_match.group(1))
                        return f"H{hour + 1}"
        except:
            pass
        return None

    def get_britned_data(self, delivery_day=None):
        if delivery_day is None:
            delivery_day = datetime.now().strftime('%Y-%m-%d')
        endpoint = f"{self.britned_api_base}/v1/public/nominations/aggregated-overview"
        params = {'deliveryDay': delivery_day, 'timescales': 'INTRA_DAY'}
        try:
            r = self.session.get(endpoint, params=params, timeout=30)
            if r.status_code == 200:
                return self.parse_britned_data(r.json())
        except:
            pass
        return None

    def get_britned_data_multi_day(self):
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        dfs = []
        for d in (today, tomorrow):
            df = self.get_britned_data(d)
            if df is not None and not df.empty: dfs.append(df)
        if not dfs: return None
        out = pd.concat(dfs, ignore_index=True)
        return out.sort_values('Time').reset_index(drop=True)

    def parse_britned_data(self, data):
        if not data or 'mtus' not in data: return None
        rows = []
        for mtu in data['mtus']:
            ts = mtu.get('mtu', '')
            if not ts: continue
            try:
                dt_utc = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                ams = pytz.timezone('Europe/Amsterdam')
                dt_local = dt_utc.astimezone(ams)
                time_period = dt_local.strftime('%H:%M')
                date_str = dt_local.strftime('%Y-%m-%d')
            except:
                continue
            time_slot = f"{date_str} {time_period}-{(pd.to_datetime(time_period, format='%H:%M') + pd.Timedelta(hours=1)).strftime('%H:%M')}"
            row = {'Time': time_slot}
            for value in mtu.get('values', []):
                direction = value.get('direction', '')
                nominations = (value.get('aggregatedNominations', 0) or 0)/1000
                if direction == 'GB_NL':
                    row['GB>NL'] = int(nominations)
                elif direction == 'NL_GB':
                    row['NL>GB'] = int(nominations)
            rows.append(row)
        return pd.DataFrame(rows)

    def get_all_data(self, delivery_day=None, use_selenium=False):
        britned_df = self.get_britned_data_multi_day()
        if britned_df is None:
            return pd.DataFrame({'Date': [], 'Hour': [], 'GB>NL': [], 'NL>GB': [], 'Total': []})
        britned_df['Date'] = britned_df['Time']
        britned_df['Hour'] = britned_df['Date'].apply(self.extract_hour_from_time)
        cols = ['Date', 'Hour']
        data_cols = [c for c in britned_df.columns if c not in ['Date', 'Hour', 'Time']]
        britned_df = britned_df[cols + data_cols]
        gb_in = [c for c in britned_df.columns if '>GB' in c]
        gb_out = [c for c in britned_df.columns if 'GB>' in c]
        total_in = britned_df[gb_in].sum(axis=1) if gb_in else 0
        total_out = britned_df[gb_out].sum(axis=1) if gb_out else 0
        britned_df['Total'] = total_in - total_out
        return britned_df

def get_interconnector_data():
    try:
        df = AllInOneInterconnectorFetcher().get_all_data()
        return df.to_string(index=False) if df is not None and len(df)>0 else "No interconnector data available"
    except Exception as e:
        return f"Interconnector error: {str(e)}"

# ===== NETHERLANDS MERIT ORDER =====
def get_nl_merit_order():
    try:
        api_key = '41ca175f-af99-4b85-a933-94fe16b453cf'
        now = datetime.now()
        rounded_now = now.replace(second=0, microsecond=0, minute=now.minute // 15 * 15)
        next_interval = rounded_now + timedelta(minutes=600)
        date_from = rounded_now.strftime('%d-%m-%Y %H:%M:%S')
        date_to   = next_interval.strftime('%d-%m-%Y %H:%M:%S')
        url = f"https://api.tennet.eu/publications/v1/merit-order-list?date_from={date_from}&date_to={date_to}"
        r = requests.get(url, headers={'apikey': api_key, 'Accept':'application/json'}, timeout=30)
        if r.status_code != 200:
            return f"Failed to retrieve NL merit order data. Status code: {r.status_code}"
        data = r.json()
        ts = data['Response']['TimeSeries']
        df_list = []
        for entry in ts:
            for point in entry['Period']['Points']:
                thresholds = pd.json_normalize(point['Thresholds'])
                thresholds['timeInterval_start'] = point['timeInterval_start']
                thresholds['timeInterval_end'] = point['timeInterval_end']
                thresholds['isp'] = point['isp']
                thresholds['mRID'] = entry['mRID']
                thresholds['quantity_measurement_unit_name'] = entry['quantity_Measurement_Unit_name']
                thresholds['price_measurement_unit_name'] = entry['price_Measurement_Unit_name']
                thresholds['currency_unit_name'] = entry['currency_Unit_name']
                df_list.append(thresholds)
        df = pd.concat(df_list, ignore_index=True)
        cet = pytz.timezone('Europe/Amsterdam')
        df['timeInterval_start'] = pd.to_datetime(df['timeInterval_start']).dt.tz_localize(cet)
        df['timeInterval_end']   = pd.to_datetime(df['timeInterval_end']).dt.tz_localize(cet)
        df.rename(columns={
            'timeInterval_start':'Date','timeInterval_end':'Date_End',
            'capacity_threshold':'Volume','price_down':'Price Down',
            'price_up':'Price Up','isp':'ISP'
        }, inplace=True)
        df = df[['Date','ISP','Volume','Price Down','Price Up']].drop_duplicates().reset_index(drop=True)
        for c in ['Volume','Price Down','Price Up']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.sort_values(['Date','Volume']).reset_index(drop=True)
        df['Price Down'] = df.groupby('ISP')['Price Down'].fillna(method='ffill')
        df['Price Up']   = df.groupby('ISP')['Price Up'].fillna(method='ffill')
        df['Time'] = df['Date'].dt.strftime('%H:%M')
        desired_thresholds = [300,250,200,150,100,50]
        proc = []
        for _, row in df.iterrows():
            if row['Volume'] in desired_thresholds:
                if pd.notna(row['Price Down']):
                    proc.append({'Time':row['Time'],'Volume':f"-{row['Volume']:.0f}",'Price':row['Price Down']})
                if pd.notna(row['Price Up']):
                    proc.append({'Time':row['Time'],'Volume':f"{row['Volume']:.0f}",'Price':row['Price Up']})
        proc_df = pd.DataFrame(proc)
        if proc_df.empty: return "No price data available for the requested time period."
        pivot = proc_df.pivot_table(index='Time', columns='Volume', values='Price', aggfunc='first')
        desired_cols = ['-300','-250','-200','-150','-100','-50','50','100','150','200','250','300']
        avail = [c for c in desired_cols if c in pivot.columns]
        if not avail: return "No data available for the standard volume thresholds."
        final = pivot[avail].reset_index().rename(columns={'Time':'Date'})
        num = final.select_dtypes(include=[np.number]).columns
        final[num] = final[num].round(2)
        final = final.sort_values('Date')
        return final.to_string(index=False, float_format='%.2f')
    except Exception as e:
        return f"NL Merit Order error: {str(e)}"

# ===== NETHERLANDS BALANCE DELTA =====
def get_nl_balance_delta():
    try:
        api_key = '41ca175f-af99-4b85-a933-94fe16b453cf'
        now = datetime.now() - timedelta(minutes=10)
        rounded_now = now.replace(second=0, microsecond=0, minute=now.minute // 15 * 15)
        next_interval = rounded_now + timedelta(minutes=60)
        date_from = rounded_now.strftime('%d-%m-%Y %H:%M:%S')
        date_to   = next_interval.strftime('%d-%m-%Y %H:%M:%S')
        url = f"https://api.tennet.eu/publications/v1/balance-delta?date_from={date_from}&date_to={date_to}"
        r = requests.get(url, headers={'apikey': api_key, 'Accept':'application/json'}, timeout=30)
        if r.status_code != 200:
            return f"Failed to fetch NL balance delta data. Status code: {r.status_code}"
        data = r.json()
        df_list = []
        for entry in data['Response']['TimeSeries']:
            for point in entry['Period']['Points']:
                df_list.append({
                    'timeInterval_start': point.get('timeInterval_start'),
                    'timeInterval_end': point.get('timeInterval_end'),
                    'power_afrr_in': point.get('power_afrr_in', None),
                    'power_afrr_out': point.get('power_afrr_out', None),
                    'power_igcc_in': point.get('power_igcc_in', None),
                    'power_igcc_out': point.get('power_igcc_out', None),
                    'power_mfrrda_in': point.get('power_mfrrda_in', None),
                    'power_mfrrda_out': point.get('power_mfrrda_out', None),
                    'power_picasso_in': point.get('power_picasso_in', None),
                    'power_picasso_out': point.get('power_picasso_out', None),
                    'max_upw_regulation_price': point.get('max_upw_regulation_price', None),
                    'min_downw_regulation_price': point.get('min_downw_regulation_price', None),
                    'mid_price': point.get('mid_price', None)
                })
        df = pd.DataFrame(df_list)
        df['timeInterval_start'] = pd.to_datetime(df['timeInterval_start']).dt.tz_localize('Europe/Amsterdam')
        df['Time'] = df['timeInterval_start'].dt.strftime('%H:%M')
        df.fillna(0.0, inplace=True)
        num_cols = ['power_afrr_in','power_afrr_out','power_igcc_in','power_igcc_out',
                    'power_mfrrda_in','power_mfrrda_out','power_picasso_in','power_picasso_out',
                    'max_upw_regulation_price','min_downw_regulation_price','mid_price']
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        final_df = pd.DataFrame({
            'Date Start': df['Time'],
            'Power AFRR out': df['power_afrr_out'],
            'Power IGCC out': df['power_igcc_out'],
            'Power Picasso in': df['power_picasso_in'],
            'Power Picasso out': df['power_picasso_out'],
            'Min DownW Regulation Price': df['min_downw_regulation_price'],
            'Mid Price': df['mid_price']
        }).sort_values('Date Start', ascending=False).reset_index(drop=True)
        ncols = final_df.select_dtypes(include=[np.number]).columns
        final_df[ncols] = final_df[ncols].round(1)
        return final_df.to_string(index=False, float_format='%.1f')
    except Exception as e:
        return f"NL Balance Delta error: {str(e)}"

# ===== AUSTRIAN MERIT ORDER =====
def get_at_merit_order():
    try:
        today_date = datetime.now().strftime("%Y-%m-%dT000000")
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT000000")
        posnegs = ["POS","NEG"]
        df_list = []
        for posneg in posnegs:
            url = f"https://transparency.apg.at/api/v1/SET/Data/English/PT15M/{today_date}/{tomorrow_date}/Full?p_setMode=Overview&p_setFilterRRType=SRR&p_setFilterRRDirection={posneg}"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            try:
                bids = resp.json()['ResponseData']['CustomData']['SingleBids']
                recs = [{
                    'ProductIdentifier': p.get('ProductIdentifier'),
                    'FromUtc': p.get('FromUtc'),
                    'ToUtc': p.get('ToUtc'),
                    'OfferedCapacity_MW': p.get('OfferedCapacity_MW'),
                    'EnergyPrice_EURMWH': p.get('EnergyPrice_EURMWH'),
                    'Rank': p.get('Rank'),
                    'Direction': posneg
                } for p in bids]
                df_list.append(pd.DataFrame(recs))
            except KeyError:
                continue
        if not df_list: return "No Austrian merit order data available"
        df_final = pd.concat(df_list, ignore_index=True)
        current_time = pd.Timestamp(datetime.utcnow(), tz='UTC') - timedelta(minutes=30)
        df_final['FromUtc'] = pd.to_datetime(df_final['FromUtc'])
        if df_final['FromUtc'].dt.tz is None:
            df_final['FromUtc'] = df_final['FromUtc'].dt.tz_localize('UTC')
        df_final = df_final[df_final['FromUtc'] >= current_time]
        df_final = df_final.sort_values(['ProductIdentifier','Rank'])
        df_final['RunningSum_OfferedCapacity_MW'] = df_final.groupby('ProductIdentifier')['OfferedCapacity_MW'].cumsum()
        thresholds = [50,100,150,200,250,300,500,750]
        results = []
        for pid, sub in df_final.groupby('ProductIdentifier'):
            sub = sub.sort_values(['ProductIdentifier','Rank'])
            if not sub.empty: results.append(sub.iloc[-1])
            for th in thresholds:
                hit = sub[sub['RunningSum_OfferedCapacity_MW'] >= th]
                if not hit.empty:
                    tmp = hit.iloc[0].copy()
                    tmp['threshold'] = th
                    results.append(tmp)
        df = pd.DataFrame(results).sort_values(['ProductIdentifier','FromUtc','Rank'])
        df['FromUtc'] = pd.to_datetime(df['FromUtc'], utc=True).dt.tz_convert('CET')
        def calc_product(dt):
            hour_plus_one = (dt.hour) % 24 + 1
            quarter = dt.minute // 15 + 1
            return f"{dt.date()} - {hour_plus_one}.{quarter}"
        df['FromUtc'] = df['FromUtc'].apply(calc_product)
        conditions = [
            (df['Direction']=='NEG') & (pd.notna(df.get('threshold'))),
            (df['Direction']=='NEG') & (pd.isna(df.get('threshold'))),
            (df['Direction']=='POS') & (pd.isna(df.get('threshold'))),
            (df['Direction']=='POS') & (pd.notna(df.get('threshold')))
        ]
        choices = [
            '-' + df['threshold'].astype(str),
            'MIN', 'MAX',
            df['threshold'].astype(str)
        ]
        df['threshold'] = np.select(conditions, choices)
        pivot = df.pivot_table(index='FromUtc', columns='threshold',
                               values='EnergyPrice_EURMWH', aggfunc='first')
        desired = ['MIN','-750.0','-500.0','-300.0','-250.0','-200.0','-150.0','-100.0','-50.0',
                   '50.0','100.0','150.0','200.0','250.0','300.0','500.0','750.0','MAX']
        pivot = pivot[[c for c in desired if c in pivot.columns]]
        final = pivot.reset_index().rename(columns={'FromUtc':'Date'})
        num = final.select_dtypes(include=[np.number]).columns
        final[num] = final[num].round(2)
        return final.to_string(index=False, float_format='%.2f')
    except Exception as e:
        return f"Austrian Merit Order error: {str(e)}"

# ===== AUSTRIAN CROSS ZONAL CAPACITIES =====
def fetch_apg_data(target_date, border):
    base_url = "https://transparency.apg.at/api/v1"
    from_date = target_date.strftime('%Y-%m-%d') + 'T000000'
    to_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d') + 'T000000'
    url = f"{base_url}/ATC/Download/English/PT15M/{from_date}/{to_date}?p_border={border}"
    try:
        r = requests.get(url, timeout=45)
        if r.status_code == 200: return r.text
    except:
        pass
    return None

def clean_and_parse_csv(csv_content):
    if not csv_content: return None
    csv_content = csv_content.replace('\ufeff', '').replace('ï»¿', '')
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        df['Time from [CET/CEST]'] = pd.to_datetime(df['Time from [CET/CEST]'])
        df['Time to [CET/CEST]']   = pd.to_datetime(df['Time to [CET/CEST]'])
        return df
    except:
        return None

def create_unified_table_for_date(target_date):
    borders = {'AT<>CH':'CH','AT<>DE':'DE','AT<>CZ':'CZ','AT<>HU':'HU','AT<>SI':'SI','AT<>IT':'IT','AT<>SK':'SK'}
    all_data = []
    for border_code, cc in borders.items():
        csv_data = fetch_apg_data(target_date, border_code)
        df = clean_and_parse_csv(csv_data)
        if df is None or df.empty: continue
        df['DateTime'] = df['Time from [CET/CEST]'].dt.strftime('%d-%m-%Y')
        def calc_product(dt):
            hour_plus_one = (dt.hour) % 24 + 1
            quarter = dt.minute // 15 + 1
            return f"{hour_plus_one}.{quarter}"
        df['Product'] = df['Time from [CET/CEST]'].apply(calc_product)
        at_to_foreign = [c for c in df.columns if c.startswith('Offered Capacities AT>')][0]
        foreign_to_at = [c for c in df.columns if c.startswith('Offered Capacities') and '>AT' in c][0]
        for _, row in df.iterrows():
            all_data.append({
                'Date': row['DateTime'],
                'Product': row['Product'],
                f'AT>{cc}': round(row[at_to_foreign], 1),
                f'{cc}>AT': round(row[foreign_to_at], 1),
                'datesort': row['Time from [CET/CEST]']
            })
    if not all_data: return None
    result_df = pd.DataFrame(all_data)
    grouped = result_df.groupby(['Date','Product']).first().reset_index()

    at_out_columns = [col for col in grouped.columns if col.startswith('AT>')]
    at_in_columns  = [col for col in grouped.columns if col.endswith('>AT')]

    def get_isolation_status(row):
        is_down = pd.to_numeric(row[at_out_columns], errors='coerce').fillna(0).sum() == 0
        is_up   = pd.to_numeric(row[at_in_columns], errors='coerce').fillna(0).sum() == 0
        if is_down and is_up: return "FULL"
        elif is_down: return "↓"
        elif is_up: return "↑"
        else: return ""

    grouped['Isolated'] = grouped.apply(get_isolation_status, axis=1)
    cols = grouped.columns.tolist()
    cols.insert(2, cols.pop(cols.index('Isolated')))
    grouped = grouped[cols]
    return grouped

def get_at_cross_zonal_caps():
    try:
        today = date.today()
        df = None
        for attempt in range(3):
            df = create_unified_table_for_date(today)
            if df is not None and df.shape[1] >= 10: break
            time.sleep(10)
        if df is None: return "No Austrian cross zonal data available"

        now = datetime.now()
        minute = (now.minute // 15 + 1) * 15
        if minute == 60:
            rounded_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            rounded_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
        df = df.sort_values(by='datesort')
        df = df[df['datesort'] >= rounded_time]
        df = df.drop(columns='datesort')

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%d-%m')
        df.rename(columns={'Date':'Day','Product':'PD'}, inplace=True)

        value_cols = [c for c in df.columns if c not in ['Day','PD','Isolated']]

        for i, row in df.iterrows():
            isolated = str(row.get('Isolated') or '').strip()
            at_gt_cols = [c for c in df.columns if 'AT>' in c]
            gt_at_cols = [c for c in df.columns if '>AT' in c]
            at_gt_sum = sum(pd.to_numeric([row.get(c,'')], errors='coerce').fillna(0) for c in at_gt_cols)
            gt_at_sum = sum(pd.to_numeric([row.get(c,'')], errors='coerce').fillna(0) for c in gt_at_cols)

            if isolated:
                for col in (value_cols + ['Isolated']):
                    value = row.get(col, '')
                    if isolated == '↓' and ('AT>' in col or col == 'Isolated'):
                        df.at[i, col] = f'🔻 {value}'.strip()
                    elif isolated == '↑' and ('>AT' in col or col == 'Isolated'):
                        df.at[i, col] = f'🔺 {value}'.strip()
                    elif isolated not in ('↑','↓'):
                        df.at[i, col] = f'🟥 {value}'.strip()

            for col in at_gt_cols:
                value = df.at[i, col]
                try:
                    if 0 < float(str(at_gt_sum)) < 201:
                        df.at[i, col] = f'🟡 {value}'.strip() if value else '🟡'
                except: pass
            for col in gt_at_cols:
                value = df.at[i, col]
                try:
                    if 0 < float(str(gt_at_sum)) < 201:
                        df.at[i, col] = f'🟡 {value}'.strip() if value else '🟡'
                except: pass

        return df.to_string(index=False)
    except Exception as e:
        return f"Austrian Cross Zonal error: {str(e)}"

# ===== HTML DASHBOARD =====
def create_comprehensive_dashboard(data):
    def format_section_content(content):
        if isinstance(content, str):
            if content.startswith("Error"):
                return f'<div class="error">{content}</div>'
            else:
                return f'<pre class="data-output">{content}</pre>'
        else:
            return f'<pre class="data-output">{str(content)}</pre>'
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Energy Trading Dashboard</title>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                margin: 0; padding: 15px; background: #0d1117; color: #c9d1d9;
                font-size: 11px; line-height: 1.3;
            }}
            .dashboard {{ max-width: 1600px; margin: 0 auto; }}
            .header {{
                background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center;
                border: 1px solid #374151; box-shadow: 0 4px 6px rgba(0,0,0,.3);
            }}
            .header h1 {{ color: #60a5fa; margin: 0 0 10px; font-size: 24px; font-weight: bold; }}
            .header p {{ color: #9ca3af; margin: 0; font-size: 14px; }}
            .section {{ background: #161b22; margin-bottom: 20px; border-radius: 6px; border: 1px solid #30363d; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,.2); }}
            .section-header {{ background: #21262d; padding: 12px 16px; border-bottom: 1px solid #30363d; font-weight: bold; color: #58a6ff; font-size: 14px; display:flex; align-items:center; gap:8px; }}
            .data-output {{ margin:0; padding:16px; background:#0d1117; color:#c9d1d9; font-family:'Consolas','Monaco','Courier New',monospace; font-size:11px; line-height:1.2; white-space:pre; overflow-x:auto; border:none; border-radius:0; }}
            .error {{ color:#f85149; background:#0d1117; padding:16px; border-left:4px solid #f85149; font-family:'Consolas','Monaco','Courier New',monospace; font-size:11px; }}
            .timestamp {{ text-align:center; color:#6e7681; margin:30px 0 20px; font-style:italic; font-size:12px; padding:10px; background:#161b22; border-radius:6px; border:1px solid #30363d; }}
            .footer-links {{ text-align:center; margin-top:20px; padding:15px; background:#161b22; border-radius:6px; border:1px solid #30363d; }}
            .footer-links a {{ color:#58a6ff; text-decoration:none; padding:8px 16px; border:1px solid #30363d; border-radius:4px; transition:all .2s; display:inline-block; font-size:12px; }}
            .footer-links a:hover {{ background:#21262d; border-color:#58a6ff; }}
            ::-webkit-scrollbar {{ width:8px; height:8px; }} ::-webkit-scrollbar-track {{ background:#161b22; }}
            ::-webkit-scrollbar-thumb {{ background:#30363d; border-radius:4px; }} ::-webkit-scrollbar-thumb:hover {{ background:#484f58; }}
            @media (max-width:768px) {{
                body {{ padding:10px; font-size:10px; }}
                .header h1 {{ font-size:20px; }}
                .data-output {{ font-size:10px; padding:12px; }}
                .section-header {{ font-size:13px; padding:10px 12px; }}
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>⚡ Energy Trading Dashboard</h1>
                <p>Real-time market data and interconnector nominations</p>
            </div>
            <div class="section"><div class="section-header">🔌 UK Interconnector Nominations (BritNed & RNP)</div>
                {format_section_content(data.get('interconnector', 'No data available'))}
            </div>
            <div class="section"><div class="section-header">🇳🇱 Netherlands Merit Order List (TenneT)</div>
                {format_section_content(data.get('nl_merit_order', 'No data available'))}
            </div>
            <div class="section"><div class="section-header">🇳🇱 Netherlands Balance Delta (TenneT)</div>
                {format_section_content(data.get('nl_balance_delta', 'No data available'))}
            </div>
            <div class="section"><div class="section-header">🇦🇹 Austrian Merit Order (APG)</div>
                {format_section_content(data.get('at_merit_order', 'No data available'))}
            </div>
            <div class="section"><div class="section-header">🇦🇹 Austrian Cross Zonal Capacities (APG)</div>
                {format_section_content(data.get('at_cross_zonal', 'No data available'))}
            </div>
            <div class="timestamp">🕐 Last Updated: {data.get('timestamp', 'Unknown')}</div>
            <div class="footer-links"><a href="?format=json">📄 View Raw JSON Data</a></div>
        </div>
    </body>
    </html>
    """
    return html
