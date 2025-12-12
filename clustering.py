results = []

# all sectors in the pivot
sectors = df_transact_by_sector_custom.columns.get_level_values(0).unique()

for sector in sectors:
    # safely skip sectors that don't have both DEBIT and CREDIT
    if 'DEBIT' not in df_transact_by_sector_custom[sector].columns or \
       'CREDIT' not in df_transact_by_sector_custom[sector].columns:
        continue

    debit  = df_transact_by_sector_custom[sector]['DEBIT']
    credit = df_transact_by_sector_custom[sector]['CREDIT']

    # put the two together for easy alignment later
    sector_df = pd.concat(
        [debit.rename('DEBIT'), credit.rename('CREDIT')],
        axis=1
    )

    for macro_name in macro_df.columns:
        # align on common dates and drop NaNs
        merged = pd.concat(
            [sector_df, macro_df[macro_name].rename(macro_name)],
            axis=1,
            join='inner'
        ).dropna()

        if len(merged) < 3:   # not enough points to compute correlation robustly
            continue

        X1 = merged['DEBIT']
        X2 = merged['CREDIT']
        Y  = merged[macro_name]

        w_debit, w_credit, corr = maximize_correlation(X1, X2, Y)

        results.append({
            'sector'           : sector,
            'macro_indicator'  : macro_name,
            'corr_with_Y'      : corr,
            'weight_debit'     : w_debit,
            'weight_credit'    : w_credit
        })

results_df = pd.DataFrame(results)
