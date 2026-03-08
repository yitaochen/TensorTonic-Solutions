import math 

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    n = len(production_log)
    accuracy_prod = sum(production_log[i]["prediction"]==production_log[i]["actual"] for i in range(n)) / n
    accuracy_shadow = sum(shadow_log[i]["prediction"]==shadow_log[i]["actual"] for i in range(n)) / n
    accuracy_gain = accuracy_shadow - accuracy_prod
    index = math.ceil(0.95 * n) - 1
    latencies_shadow = sorted(shadow_log[i]["latency_ms"] for i in range(n))
    shadow_latency_p95 = latencies_shadow[index]
    agreement_rate = sum(production_log[i]["prediction"]==shadow_log[i]["prediction"] for i in range(n)) / n
    promote = False 
    if accuracy_gain >= criteria["min_accuracy_gain"] and shadow_latency_p95 <= criteria["max_latency_p95"] and agreement_rate >= criteria["min_agreement_rate"]:
        promote = True 
    return {"promote": promote, "metrics": {"shadow_accuracy": accuracy_shadow, "production_accuracy": accuracy_prod, "accuracy_gain": accuracy_gain, "shadow_latency_p95": shadow_latency_p95, "agreement_rate": agreement_rate}}