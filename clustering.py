import matplotlib.pyplot as plt

def plot_yoy_series(sector, macro_name, merged_yoy, w_debit, w_credit):
    """
    merged_yoy: DataFrame with columns ['X1','X2','Y']
    """

    # Compute optimized combined YoY series
    merged_yoy['Opt_Combined'] = w_debit * merged_yoy['X1'] + w_credit * merged_yoy['X2']

    plt.figure(figsize=(12,6))
    plt.plot(merged_yoy.index.to_timestamp(), merged_yoy['X1'], label='DEBIT YoY', linewidth=2)
    plt.plot(merged_yoy.index.to_timestamp(), merged_yoy['X2'], label='CREDIT YoY', linewidth=2)
    plt.plot(merged_yoy.index.to_timestamp(), merged_yoy['Y'], label=f'{macro_name} YoY', linewidth=2)
    plt.plot(merged_yoy.index.to_timestamp(), merged_yoy['Opt_Combined'], 
             label=f'Optimized ({w_debit:.2f},{w_credit:.2f})', 
             linewidth=3, linestyle='--')
    
    plt.title(f"YoY Series: {sector} vs {macro_name}")
    plt.xlabel("Date")
    plt.ylabel("YoY % Change")
    plt.legend()
    plt.grid(True)
    plt.show()
