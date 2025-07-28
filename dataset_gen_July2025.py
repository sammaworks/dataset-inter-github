import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

def generate_synthetic_security_alerts(
    NUM_ALERTS=8000,
    START_DATE=datetime(2023, 1, 1),
    label_noise=0.10,      # 10% random label flips
    feature_noise=0.50,    # 50% jitter on all continuous features
    random_seed=42
) -> pd.DataFrame:
    """
    Generates a synthetic security alerts dataset with:
     - realistic timestamp burstiness
     - label noise
     - heavy feature overlap
     - realistic, skewed & multimodal CPU/memory usage distributions
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

    # 2) Draw incident priorities & inject label noise
    priors = ['Low','Medium','High','Critical']
    base_probs = [0.75, 0.20, 0.04, 0.01]
    incident_priority = np.random.choice(priors, size=NUM_ALERTS, p=base_probs)
    flip = np.random.rand(NUM_ALERTS) < label_noise
    incident_priority[flip] = np.random.choice(priors, size=flip.sum(), p=base_probs)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'incident_priority': incident_priority
    })

    # Helper: sample per-category + jitter
    def sample_cond(cats, mapping):
        arr = np.empty(len(cats), dtype=float)
        for cat, fn in mapping.items():
            idx = np.where(cats == cat)[0]
            arr[idx] = fn(len(idx))
        arr += np.random.normal(0, np.nanstd(arr) * feature_noise, size=len(arr))
        return arr

    # 3) CVE score ∈ [0,10]
    cve_map = {
        'Low':      lambda n: np.random.beta(1,2,n)*5,
        'Medium':   lambda n: np.random.beta(2,2,n)*7,
        'High':     lambda n: np.random.beta(2,1,n)*8 + 1,
        'Critical': lambda n: np.random.beta(3,1,n)*9 + 1
    }
    df['cve_score'] = sample_cond(df['incident_priority'], cve_map).clip(0,10).round(1)

    # 4) IP reputation ∈ [0,100]
    rep_map = {
        'Low':      lambda n: np.random.beta(3,1,n)*80,
        'Medium':   lambda n: np.random.beta(2,2,n)*80 + 10,
        'High':     lambda n: np.random.beta(1,2,n)*80 + 20,
        'Critical': lambda n: np.random.beta(1,3,n)*70 + 30
    }
    df['ip_reputation_score'] = sample_cond(df['incident_priority'], rep_map).clip(0,100).round(1)

    # 5) Login attempts
    login_map = {
        'Low':      lambda n: np.random.poisson(1, n),
        'Medium':   lambda n: np.random.poisson(5, n),
        'High':     lambda n: np.random.poisson(15,n),
        'Critical': lambda n: np.random.poisson(30,n)
    }
    df['login_attempts'] = sample_cond(df['incident_priority'], login_map).clip(0).round().astype(int)

    # 6) CPU & memory usage: skewed + diurnal + spikes
    # diurnal effect
    hours = np.array([ts.hour for ts in df['timestamp']])
    diurnal = 10 * np.sin(2 * np.pi * hours / 24) + 10  # ranges approx 0–20

    # base samplers
    def cpu_base(cat, n):
        params = {'Low':(2,5), 'Medium':(4,4), 'High':(5,2), 'Critical':(2,1)}
        a,b = params[cat]
        return np.random.beta(a,b,n)*100
    def mem_base(cat, n):
        params = {'Low':(3,4), 'Medium':(5,3), 'High':(6,2), 'Critical':(7,1)}
        a,b = params[cat]
        return np.random.beta(a,b,n)*100

    cpu = np.zeros(NUM_ALERTS)
    mem = np.zeros(NUM_ALERTS)
    for cat in priors:
        idx = np.where(df['incident_priority']==cat)[0]
        cpu[idx] = cpu_base(cat, len(idx))
        mem[idx] = mem_base(cat, len(idx))

    # inject random spikes
    spike_prob = {'Low':0.01,'Medium':0.02,'High':0.05,'Critical':0.10}
    spike_amt = {'Low':(5,15),'Medium':(10,30),'High':(20,50),'Critical':(30,70)}
    for i, p in enumerate(df['incident_priority']):
        if np.random.rand() < spike_prob[p]:
            inc = np.random.uniform(*spike_amt[p])
            cpu[i] = min(100, cpu[i] + inc)
        if np.random.rand() < spike_prob[p]:
            inc = np.random.uniform(*spike_amt[p])
            mem[i] = min(100, mem[i] + inc)

    # combine base + diurnal + jitter
    cpu = cpu + diurnal + np.random.normal(0, np.nanstd(cpu)*feature_noise, NUM_ALERTS)
    mem = mem + diurnal + np.random.normal(0, np.nanstd(mem)*feature_noise, NUM_ALERTS)
    df['cpu_usage_percent']    = np.clip(cpu,0,100).round(1)
    df['memory_usage_percent'] = np.clip(mem,0,100).round(1)

    # 7) Payload size (bytes)
    def payload_sampler(cat,n):
        base = {'Low':800,'Medium':1200,'High':2000,'Critical':2500}[cat]
        return np.random.lognormal(mean=np.log(base), sigma=0.8, size=n)
    pay = np.zeros(NUM_ALERTS)
    for cat in priors:
        idx = np.where(df['incident_priority']==cat)[0]
        pay[idx] = payload_sampler(cat,len(idx))
    pay += np.random.normal(0, np.nanstd(pay)*feature_noise, NUM_ALERTS)
    pay = np.clip(pay, 0, None)
    df['payload_size'] = pay.round().astype(int)

    # 8) Contextual fields
    df['source_ip'] = [fake.ipv4() for _ in range(NUM_ALERTS)]
    df['user_role'] = np.random.choice(
        ['Admin','Developer','Finance','Support','Contractor'],
        p=[0.1,0.3,0.2,0.25,0.15], size=NUM_ALERTS
    )
    df['system_context'] = np.random.choice(
        ['Database','Web Server','Endpoint','API Gateway','Cloud Instance'],
        p=[0.2,0.3,0.2,0.2,0.1], size=NUM_ALERTS
    )

    # 9) Outcome & remediation time
    outcome_map = {'Low':'Blocked','Medium':'Investigated','High':'Quarantined','Critical':'Escalated'}
    df['outcome'] = df['incident_priority'].map(outcome_map)
    df['time_to_remediate_hours'] = np.random.exponential(
        scale=df['cve_score']*2 + 1, size=NUM_ALERTS
    ).round(1)

    # 10) Shuffle and return
    return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)


if __name__ == '__main__':
    df = generate_synthetic_security_alerts()
    df.to_csv('synthetic_security_alerts.csv', index=False)
    print("Generated challenging dataset with shape:", df.shape)
    print(df['incident_priority'].value_counts(normalize=True))
