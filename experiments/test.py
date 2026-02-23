import pulp

# ==========================================
# 1. SETUP & DATA
# ==========================================
P_xy = 0.4010  # P(X=1, Y=1)
P_xy_p = 0.0960  # P(X=1, Y=0)
P_xp_y = 0.1530  # P(X=0, Y=1)
P_xp_yp = 0.3500  # P(X=0, Y=0)

assert P_xy + P_xy_p + P_xp_y + P_xp_yp == 1

print(
    f"Observational Data: P(1,1)={P_xy}, P(1,0)={P_xy_p}, P(0,1)={P_xp_y}, P(0,0)={P_xp_yp}"
)

# ==========================================
# 2. SOLVE LP (Finding the Optimal q*)
# ==========================================
# Variables p_ijk correspond to the 8 Canonical Worlds
# i = Y_{x=1} (Response to X=1)
# j = Y_{x=0} (Response to X=0)
# k = X       (Initial X)

# We want to MINIMIZE PNS to find the Lower Bound
prob = pulp.LpProblem("Tightness_Proof", pulp.LpMinimize)

# Decision Variables (The q weights for the 8 worlds)
p111 = pulp.LpVariable("p111", lowBound=0)
p110 = pulp.LpVariable("p110", lowBound=0)  # Worlds where X=1
p101 = pulp.LpVariable("p101", lowBound=0)
p100 = pulp.LpVariable("p100", lowBound=0)
p011 = pulp.LpVariable("p011", lowBound=0)
p010 = pulp.LpVariable("p010", lowBound=0)
p001 = pulp.LpVariable("p001", lowBound=0)
p000 = pulp.LpVariable("p000", lowBound=0)  # Worlds where X=0

# Constraints (Observational Consistency)
prob += p111 + p101 == P_xy  # Consistent with X=1, Y=1
prob += p011 + p001 == P_xy_p  # Consistent with X=1, Y=0
prob += p110 + p010 == P_xp_y  # Consistent with X=0, Y=1
prob += p100 + p000 == P_xp_yp  # Consistent with X=0, Y=0

# Objective: PNS = P(Y_{x=1}=1, Y_{x=0}=0)
# This corresponds to worlds where i=1 and j=0
prob += p101 + p100

prob.solve(pulp.PULP_CBC_CMD(msg=False))

# Extract the optimal distribution q* over the 8 worlds
q_star = {
    "p111": pulp.value(p111),
    "p110": pulp.value(p110),
    "p101": pulp.value(p101),
    "p100": pulp.value(p100),
    "p011": pulp.value(p011),
    "p010": pulp.value(p010),
    "p001": pulp.value(p001),
    "p000": pulp.value(p000),
}

print("\n--- 1. Optimal LP Solution (q*) ---")
print(f"Calculated Lower Bound for PNS: {pulp.value(prob.objective):.6f}")
print("Non-zero world probabilities:")
for k, v in q_star.items():
    if v > 1e-6:
        print(f"  {k}: {v:.6f}")

# ==========================================
# 3. CONSTRUCT THE TIGHT SCM (M*)
# ==========================================
print("\n--- 2. Constructing SCM M* (Tightness Theorem) ---")

# Define Exogenous Variable U
# Domain: The 8 keys of q_star (p111, p110, ...)
# Distribution: P(U) = q_star values


# Structural Equation f_X(u)
# In notation p_ijk, k is X.
def f_X(u_state):
    # key format 'pijk' -> k is index 3
    k = int(u_state[3])
    return k


# Structural Equation f_Y(x, u)
# In notation p_ijk:
# i (index 1) is Y if X=1
# j (index 2) is Y if X=0
def f_Y(x_input, u_state):
    i = int(u_state[1])
    j = int(u_state[2])
    if x_input == 1:
        return i
    else:
        return j


# ==========================================
# 4. VERIFY THE SCM
# ==========================================
print("\n--- 3. Verifying SCM M* ---")

# A. Check Observational Distribution P(X, Y)
check_P_xy = 0.0
for u, prob_u in q_star.items():
    # Simulate the SCM for this u
    x_val = f_X(u)
    y_val = f_Y(x_val, u)

    if x_val == 1 and y_val == 1:
        check_P_xy += prob_u

print(f"SCM Generated P(X=1, Y=1): {check_P_xy:.4f} (Target: {P_xy})")

# B. Check Counterfactual Query PNS
# PNS = P(Y_{x=1}=1, Y_{x=0}=0)
scm_pns = 0.0
for u, prob_u in q_star.items():
    # Counterfactual 1: Force X=1
    y_do_1 = f_Y(1, u)

    # Counterfactual 2: Force X=0
    y_do_0 = f_Y(0, u)

    # Check PNS Condition
    if y_do_1 == 1 and y_do_0 == 0:
        scm_pns += prob_u

print(f"SCM Generated PNS:         {scm_pns:.6f}")
print(f"Match with LP Bound?       {abs(scm_pns - pulp.value(prob.objective)) < 1e-6}")
