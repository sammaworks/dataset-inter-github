# ------------------------------------------------------------
# Pairwise post-hoc: Wilcoxon signed-rank across sizes (Holm)
# ------------------------------------------------------------
def holm_bonferroni(pvals, labels):
    pairs = list(range(len(pvals)))
    # sort ascending by p
    ranked = sorted(pairs, key=lambda i: pvals[i])
    adjusted = [None]*len(pvals)
    m = len(pvals)
    prev = 0.0
    for rank, i in enumerate(ranked, start=1):
        adj = min(1.0, (m - rank + 1) * pvals[i])
        # enforce monotonicity
        adj = max(adj, prev)
        adjusted[i] = adj
        prev = adj
    out = []
    for i,(lab, p, padj) in enumerate(zip(labels, pvals, adjusted)):
        out.append((lab[0], lab[1], p, padj))
    return out

pair_labels, raw_p = [], []
for a, b in combinations(methods, 2):
    # matched pairs: sizes are blocks -> use Wilcoxon on vectors across sizes
    vec_a, vec_b = pt[a].values, pt[b].values
    try:
        stat, p = stats.wilcoxon(vec_a, vec_b, zero_method='wilcox', alternative='two-sided', mode='auto')
    except ValueError:
        # all differences zero or not enough non-zero pairs
        p = 1.0
    pair_labels.append((a, b))
    raw_p.append(p)

posthoc = holm_bonferroni(raw_p, pair_labels)
posthoc_df = pd.DataFrame(posthoc, columns=['Method_A', 'Method_B', 'p_raw', 'p_Holm']).sort_values('p_Holm')

print("\n=== Pairwise Wilcoxon (sizes as blocks), Holm-corrected ===")
print(posthoc_df.to_string(index=False))

# Build a simple “is best significantly better than X?” summary
best_vs_others = posthoc_df.query("Method_A == @best_method or Method_B == @best_method").copy()
def other(m):
    return m.Method_B if m.Method_A == best_method else m.Method_A
best_vs_others['Other'] = best_vs_others.apply(other, axis=1)
best_sig = best_vs_others['p_Holm'] < 0.05
n_sig = int(best_sig.sum())
print(f"\nBest by average rank: {best_method}")
print(f"Significant (Holm, α=0.05) wins vs others: {n_sig}/{len(best_vs_others)}")
