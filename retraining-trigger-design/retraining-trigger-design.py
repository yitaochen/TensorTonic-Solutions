def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """
    # Write code here
    last_retrain_day = 1 - (config["cooldown"] + 1)
    days_since_retrain = 0
    ans = []
    for stats in daily_stats:
        retrain_trigger = False 
        days_since_retrain += 1 
        if stats["drift_score"] > config["drift_threshold"] or stats["performance"] < config["performance_threshold"] or days_since_retrain >= config["max_staleness"]:
            retrain_trigger = True 
        day = stats["day"]
        if retrain_trigger and day - last_retrain_day >= config["cooldown"] and config["budget"] >= config["retrain_cost"]:
            ans.append(day)
            days_since_retrain = 0
            last_retrain_day = day
            config["budget"] -= config["retrain_cost"]

    return ans 
