run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, averages, variant, run_ts)
        all_averages[variant] = averages