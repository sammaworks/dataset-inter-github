import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

def generate_synthetic_security_alerts(
    NUM_ALERTS=8000,
    START_DATE=datetime(2023, 1, 1),
    label_noise=0.10,       # 10% random label flips
    feature_noise=0.50,     # 50% jitter on continuous features
    login_missing=0.10,     # 10% missingness in login_attempts
    source_ip_missing=0.52, # 52% missingness in source_ip
    random_seed=42
) -> pd.DataFrame:
    """
    Generates a synthetic security alerts dataset with:
     - alert_id
     - timestamp (standard datetime)
     - incident_priority with label noise
     - alert_type
     - realistic, skewed & multimodal CPU/memory usage distributions
     - missingness in login_attempts and source_ip
     - geolocation as country
     - outcome, remediation time
    """
    np.random.seed(random_seed)
    fake = Faker()

    # 1) Timestamps with weekday burstiness
    timestamps = []
    t = START_DATE
    while len(timestamps) < NUM_ALERTS:
        lam = np.random.choice([1, 5], p=[0.7, 0.3]) if t.weekday() < 5 else 1
        t += timedelta(minutes=np.random.exponential(scale=60/lam))
        timestamps.append(t)
    timestamps = timestamps[:NUM_ALERTS]

    # 2) Incident priorities + label noise
    priors = ['Low','Medium','High','Critical']
    base_probs = [0.75, 0.20, 0.04, 0.01]
    incident_priority = np.random.choice(priors, size=NUM_ALERTS, p=base_probs)
    flip = np.random.rand(NUM_ALERTS) < label_noise
    if flip.any():
        incident_priority[flip] = np.random.choice(priors, size=flip.sum(), p=base_probs)

    # 3) Alert types
    alert_types = ['Phishing','Malware','Brute Force','DDoS','Data Exfiltration']
    alert_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
    alert_type = np.random.choice(alert_types, size=NUM_ALERTS, p=alert_probs)

    # 4) Base DataFrame
    df = pd.DataFrame({
        'alert_id':          [f"ALERT_{i:07d}" for i in range(1, NUM_ALERTS+1)],
        'timestamp':         timestamps,
        'incident_priority': incident_priority,
        'alert_type':        alert_type
    })

    mask_alert_type = np.random.rand(NUM_ALERTS) < 0.27
    df.loc[mask_alert_type, 'alert_type'] = np.nan

    # helper: sample with jitter
    def sample_cond(cats, mapping):
        arr = np.empty(len(cats), dtype=float)
        for cat, fn in mapping.items():
            idx = np.where(cats == cat)[0]
            arr[idx] = fn(len(idx))
        arr += np.random.normal(0, np.nanstd(arr)*feature_noise, size=len(arr))
        return arr

    # 5) CVE score [0,10]
    cve_map = {
        'Low':      lambda n: np.random.beta(1,2,n)*5,
        'Medium':   lambda n: np.random.beta(2,2,n)*7,
        'High':     lambda n: np.random.beta(2,1,n)*8 + 1,
        'Critical': lambda n: np.random.beta(3,1,n)*9 + 1
    }
    df['cve_score'] = sample_cond(df['incident_priority'], cve_map).clip(0,10).round(1)

    mask_cve_score = np.random.rand(NUM_ALERTS) < 0.12
    df.loc[mask_cve_score, 'cve_score'] = np.nan

    # 6) IP reputation [0,100]
    rep_map = {
        'Low':      lambda n: np.random.beta(3,1,n)*80,
        'Medium':   lambda n: np.random.beta(2,2,n)*80 + 10,
        'High':     lambda n: np.random.beta(1,2,n)*80 + 20,
        'Critical': lambda n: np.random.beta(1,3,n)*70 + 30
    }
    df['ip_reputation_score'] = sample_cond(df['incident_priority'], rep_map).clip(0,100).round(1)

    # 7) Login attempts + missingness
    login_map = {
        'Low':      lambda n: np.random.poisson(1,n),
        'Medium':   lambda n: np.random.poisson(5,n),
        'High':     lambda n: np.random.poisson(15,n),
        'Critical': lambda n: np.random.poisson(30,n)
    }
    df['login_attempts'] = sample_cond(df['incident_priority'], login_map).round().astype(int)
    df['login_attempts'] = df['login_attempts'].clip(lower=1)
    mask_login = np.random.rand(NUM_ALERTS) < login_missing
    df.loc[mask_login, 'login_attempts'] = np.nan

    # 8) CPU & Memory usage: skewed, diurnal, spikes
    # extract hour
    hours = pd.to_datetime(df['timestamp']).dt.hour.values
    diurnal = 10 * np.sin(2*np.pi*hours/24) + 10

    cpu_params = {'Low':(2,5),'Medium':(4,4),'High':(5,2),'Critical':(2,1)}
    mem_params = {'Low':(3,4),'Medium':(5,3),'High':(6,2),'Critical':(7,1)}
    cpu = np.zeros(NUM_ALERTS); mem = np.zeros(NUM_ALERTS)
    for cat in priors:
        idx = df.index[df['incident_priority']==cat]
        a,b = cpu_params[cat]
        cpu[idx] = np.random.beta(a,b,len(idx))*100
        a2,b2 = mem_params[cat]
        mem[idx] = np.random.beta(a2,b2,len(idx))*100

    # random spikes
    spike_prob = {'Low':0.01,'Medium':0.02,'High':0.05,'Critical':0.10}
    spike_amt  = {'Low':(5,15),'Medium':(10,30),'High':(20,50),'Critical':(30,70)}
    for i, cat in enumerate(df['incident_priority']):
        if np.random.rand()<spike_prob[cat]:
            cpu[i] = min(100, cpu[i] + np.random.uniform(*spike_amt[cat]))
        if np.random.rand()<spike_prob[cat]:
            mem[i] = min(100, mem[i] + np.random.uniform(*spike_amt[cat]))

    cpu = cpu + diurnal + np.random.normal(0, np.nanstd(cpu)*feature_noise, NUM_ALERTS)
    mem = mem + diurnal + np.random.normal(0, np.nanstd(mem)*feature_noise, NUM_ALERTS)
    df['cpu_usage_percent']    = np.clip(cpu,0,100).round(1)
    df['memory_usage_percent'] = np.clip(mem,0,100).round(1)

    mask_memory_usage_percent = np.random.rand(NUM_ALERTS) < 0.1
    df.loc[mask_memory_usage_percent, 'memory_usage_percent'] = np.nan

    mask_cpu_usage_percent = np.random.rand(NUM_ALERTS) < 0.08
    df.loc[mask_cpu_usage_percent, 'cpu_usage_percent'] = np.nan

    # 9) Payload size
    def payload_sampler(cat,n):
        base = {'Low':800,'Medium':1200,'High':2000,'Critical':2500}[cat]
        return np.random.lognormal(np.log(base), 0.8, size=n)
    pay = np.zeros(NUM_ALERTS)
    for cat in priors:
        idx = df.index[df['incident_priority']==cat]
        pay[idx] = payload_sampler(cat, len(idx))
    pay += np.random.normal(0, np.nanstd(pay)*feature_noise, NUM_ALERTS)
    df['payload_size'] = np.clip(pay,0,None).round().astype(int)

    mask_payload_size = np.random.rand(NUM_ALERTS) < 0.18
    df.loc[mask_payload_size, 'payload_size'] = np.nan

    # 10) Contextual fields
    # 10a) source_ip + missingness
    df['source_ip'] = [fake.ipv4() for _ in range(NUM_ALERTS)]
    mask_ip = np.random.rand(NUM_ALERTS) < source_ip_missing
    df.loc[mask_ip, 'source_ip'] = np.nan

    # 10b) user_role & system_context
    df['user_role'] = np.random.choice(
        ['Admin','Developer','Finance','Support','Contractor'],
        p=[0.1,0.3,0.2,0.25,0.15], size=NUM_ALERTS
    )
    df['system_context'] = np.random.choice(
        ['Database','Web Server','Endpoint','API Gateway','Cloud Instance'],
        p=[0.2,0.3,0.2,0.2,0.1], size=NUM_ALERTS
    )

    # 11) Geolocation as country
    df['geolocation'] = [fake.country() for _ in range(NUM_ALERTS)]

    mask_user_role = np.random.rand(NUM_ALERTS) < 0.21
    df.loc[mask_user_role, 'user_role'] = np.nan

    mask_geolocation = np.random.rand(NUM_ALERTS) < 0.09
    df.loc[mask_geolocation, 'geolocation'] = np.nan

    mask_system_context = np.random.rand(NUM_ALERTS) < 0.08
    df.loc[mask_system_context, 'system_context'] = np.nan

    # 13) Shuffle rows
    return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)


if __name__ == '__main__':
    df = generate_synthetic_security_alerts()
    df.to_csv('security_alerts_July2025.csv', index=False)
    print("Generated dataset:", df.shape)
    print(df.head())
