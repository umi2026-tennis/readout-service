import math

from scipy import stats

def ab_test(control, treatment):
    """Compare two conversion rates.

    control and treatment are (conversions, visitors) pairs, e.g. (100, 1000).
    """
    c_conv, c_n = control
    t_conv, t_n = treatment

    # Conversion rates
    p_c = c_conv / c_n
    p_t = t_conv / t_n
    diff = p_t - p_c

    # Relative lift; undefined when control has zero conversions
    lift = diff / p_c if p_c > 0 else None

    # 95% confidence interval for the difference (treatment - control)
    se = math.sqrt(p_c * (1 - p_c) / c_n + p_t * (1 - p_t) / t_n)
    z_crit = stats.norm.ppf(0.975)
    ci_95 = (diff - z_crit * se, diff + z_crit * se)

    # Two-sided p-value from a two-proportion z-test (pooled)
    p_pool = (c_conv + t_conv) / (c_n + t_n)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / c_n + 1 / t_n))
    if se_pool == 0:
        p_value = 1.0
    else:
        z = diff / se_pool
        p_value = 2 * stats.norm.sf(abs(z))

    return {
        "control_rate": p_c,
        "treatment_rate": p_t,
        "lift": lift,
        "ci_95": ci_95,
        "p_value": p_value,
    }