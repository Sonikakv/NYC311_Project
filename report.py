def generate_report(df, results_df):
    with open("outputs/level3/reports/final_report.txt", "w") as f:

        f.write("===== MODEL PERFORMANCE =====\n")
        f.write(results_df.to_string())
        f.write("\n\n")

        f.write("===== INSIGHTS =====\n")
        f.write("- Peak hours reduce resolution efficiency\n")
        f.write("- Time and location are key factors\n")
        f.write("- Some boroughs show slower resolution\n\n")

        f.write("===== IMPROVEMENTS APPLIED =====\n")
        f.write("- SMOTE for class balancing\n")
        f.write("- Hyperparameter tuning (GridSearch)\n")
        f.write("- Cross-validation for robustness\n")
        f.write("- ROC-AUC evaluation\n\n")

        f.write("===== RECOMMENDATIONS =====\n")
        f.write("- Increase staffing during peak hours\n")
        f.write("- Prioritize slow complaint categories\n")
        f.write("- Use ML model for early delay prediction\n\n")

        f.write("===== LIMITATIONS =====\n")
        f.write("- No real-time data\n")
        f.write("- Limited feature set\n")