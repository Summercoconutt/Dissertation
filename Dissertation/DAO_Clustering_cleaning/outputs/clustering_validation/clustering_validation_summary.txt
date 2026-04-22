# Clustering validation summary

- Sample size (cleaned, with outliers in data): **434**
- Sample size (outliers removed): **377**
- Rows flagged as outliers: **57**

## Feature skewness
```
                            feature  skewness            skew_note
0               log_n_unique_voters  0.145145             moderate
1      mean_robust_participation_vp  0.653697             moderate
2       std_robust_participation_vp -1.088440  likely heavy-tailed
3                 gini_voting_power -0.748469             moderate
4               whale_ratio_top1pct  0.320079             moderate
5        proposal_frequency_per_30d  1.480524  likely heavy-tailed
6                 repeat_voter_rate -0.318607             moderate
7               z_rep_pct_for_votes -1.291457  likely heavy-tailed
8           z_rep_pct_against_votes  1.892066  likely heavy-tailed
9           z_rep_pct_abstain_votes  3.255995  likely heavy-tailed
10             z_rep_choice_entropy  1.352234  likely heavy-tailed
11  z_rep_pct_aligned_with_majority -1.564358  likely heavy-tailed
```

## Highly correlated pairs (>|0.8|)
(none)

## K-means (all sample) — metrics by k
```
   k      inertia  silhouette  calinski_harabasz  davies_bouldin                               cluster_sizes  min_cluster_size  max_cluster_pct
0  2  3339.810632    0.229701          81.001304        2.103448                            {0: 138, 1: 296}               138         0.682028
1  3  2971.685124    0.212535          72.108058        1.870264                     {0: 100, 1: 269, 2: 65}                65         0.619816
2  4  2653.260212    0.195788          70.918192        1.524948               {0: 64, 1: 249, 2: 114, 3: 7}                 7         0.573733
3  5  2412.760311    0.140114          69.044875        1.599452       {0: 104, 1: 7, 2: 134, 3: 26, 4: 163}                 7         0.375576
4  6  2232.326586    0.164676          66.480173        1.562528  {0: 78, 1: 80, 2: 7, 3: 24, 4: 75, 5: 170}                 7         0.391705
```

## K-means (no outliers) — metrics by k
```
   k      inertia  silhouette  calinski_harabasz  davies_bouldin                                cluster_sizes  min_cluster_size  max_cluster_pct
0  2  2469.926279    0.167669          71.151866        2.135650                             {0: 142, 1: 235}               142         0.623342
1  3  2184.445727    0.178067          64.556639        1.835809                      {0: 109, 1: 197, 2: 71}                71         0.522546
2  4  1911.185919    0.206933          66.836815        1.596526               {0: 104, 1: 69, 2: 171, 3: 33}                33         0.453581
3  5  1745.856545    0.157033          63.534413        1.625608         {0: 33, 1: 62, 2: 109, 3: 86, 4: 87}                33         0.289125
4  6  1645.887829    0.156302          58.276580        1.652118  {0: 75, 1: 109, 2: 40, 3: 78, 4: 44, 5: 31}                31         0.289125
```

## Null benchmark (silhouette real vs null mean)
```
   k  silhouette_real    ch_real   db_real  silhouette_null_mean  ch_null_mean  db_null_mean
0  2         0.229701  81.001304  2.103448              0.292970     51.517778      1.519548
1  3         0.212535  72.108058  1.870264              0.110270     45.946539      2.470298
2  4         0.195788  70.918192  1.524948              0.107064     42.914825      2.276915
3  5         0.140114  69.044875  1.599452              0.106301     40.705761      2.182573
4  6         0.164676  66.480173  1.562528              0.103832     38.960724      2.091417
```

## Stability (pairwise ARI)
```
       pairwise_ari
count          45.0
mean            1.0
std             0.0
min             1.0
25%             1.0
50%             1.0
75%             1.0
max             1.0
```
