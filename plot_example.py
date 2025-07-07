from plots import plot_bar_comparison, plot_radar_chart

metrics_to_plot = ["gini", "hhi", "participation", "utility"]

plot_bar_comparison(final_results, metrics=metrics_to_plot)
plot_radar_chart(final_results, metrics=metrics_to_plot)
