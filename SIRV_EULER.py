import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

N = 37_000_000  # Canada Population
GAMMA = 1 / 10  # assumed average infectious period
D = 10          # days used to estimate current I
WINDOW_LEN = 7  # weekly rolling window

# Euler step size, in days
DT = 1

BETA_MIN = 0.01
BETA_MAX = 1.50

CONFIRMED_FILE = "Canada_confirmed.csv"
VACCINATION_FILE = "vaccination-coverage-map.csv"

# load confirmed case data
def load_confirmed_data():
    confirmed = pd.read_csv(CONFIRMED_FILE)

    if "Date" in confirmed.columns:
        confirmed = confirmed.rename(columns = {"Date":"date"})
    elif "Province/State" in confirmed.columns:
        confirmed = confirmed.rename(columns = {"Province/State":"date"})

    confirmed['date'] = pd.to_datetime(confirmed['date'], format ="%m/%d/%Y")

    confirmed = confirmed[["date", "Total_confirmed"]].reset_index(drop = True)

    # daily new cases
    confirmed["new_cases_raw"] = (confirmed["Total_confirmed"].diff().fillna(0))

    # 7-day moving average
    confirmed["new_cases_smooth"] = (confirmed["new_cases_raw"].rolling(7, center=True, min_periods=1).mean())

    return confirmed

# load vaccination data
def load_vaccination_data():
    vaccination = pd.read_csv(VACCINATION_FILE)

    # pruid = 1 corresponds to Canada
    vaccination = vaccination[vaccination["pruid"] == 1].copy()
    vaccination["date"] = pd.to_datetime(vaccination["week_end"])

    # V = people with at least one vaccine dose
    vaccination["vaccinated"] = pd.to_numeric(vaccination["numtotal_atleast1dose"], errors="coerce").fillna(0)

    vaccination = vaccination[["date", "vaccinated"]]

    vaccination = (vaccination.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True))

    return vaccination

# get most recently know vaccination coverage
def get_vaccination_coverage(vaccination_data, on_date):
    """
    Return vaccination coverage known on or before on_date

    coverage = people with >= 1 dose / N
    """

    known = vaccination_data[vaccination_data["date"] <= on_date]

    # before vaccination started
    if len(known) == 0:
        return 0.0

    vaccinated = float(known.iloc[-1]["vaccinated"])

    coverage = vaccinated / N

    return np.clip(coverage, 0.0, 1.0)

# Estimate S, I, R, V on a given date
def estimate_S_I_R_V(confirmed_data, vaccination_data, on_date):
    # Estimate I
    lookback_start = (on_date - pd.Timedelta(days=D + 1))
    recent = confirmed_data[(confirmed_data["date"] >= lookback_start) & (confirmed_data["date"] <= on_date)]

    daily_new = (recent["Total_confirmed"].diff().dropna())
    I = max(float(daily_new.sum()), 0.0)

    # Estimate R
    total_today = float(confirmed_data[confirmed_data["date"] == on_date]["Total_confirmed"].iloc[0])
    R = max(total_today - I, 0.0)

    # Estimate V
    vaccination_coverage = (get_vaccination_coverage(vaccination_data, on_date))
    # population not currently in I or R
    available_population = max(N - I - R, 0.0)

    # Important Assumption
    #
    # Apply national vaccination coverage to people that are not currently classified as I or R
    #
    # This avoid directly adding cumulative vaccinated people to cumulative recovered poeple, because the real
    # DATASETS CAN CONTAIN people belonging to both groups
    V = (vaccination_coverage * available_population)
    S = ((1-vaccination_coverage) * available_population)

    return S, I, R, V

# Estimate vaccination rate nu
def estimate_nu_for_window(vaccination_data, start_date):
    """
    Estimate weekly vaccination parameter nu

    dV/dt = nu * S

    nu has units of 1/day
    """

    coverage_start = (get_vaccination_coverage(vaccination_data, start_date))

    end_date = (start_date + pd.Timedelta(days=WINDOW_LEN - 1))

    coverage_end = (get_vaccination_coverage(vaccination_data, end_date))

    # change in vaccination coverage
    coverage_change = max(coverage_end - coverage_start, 0.0)

    # Average change per day
    daily_coverage_change = (coverage_change / WINDOW_LEN)

    # Approximately:
    #
    # dp/dt = nu * (1 - p)
    #
    # therefor nu = (dp/dt) / (1 - p)

    unvaccinated_fraction = (1 - coverage_start)

    if unvaccinated_fraction <= 0.0:
        return 0.0

    nu = (daily_coverage_change / unvaccinated_fraction)

    return nu

# SIRV model using Euler's Method
def run_sirv_euler(beta, gamma, nu, N, S0, I0, R0, V0, days, dt=DT):
    """
    Solve SIRV using Forward Euler method

    Equations:
    dS/dt = -beta*S*I/N - nu*S
    dI/dt = beta*S*I/N - gamma*I
    dR/dt = gamma*I
    dV/dt = nu*S
    """

    # Number of Euler stpes per day
    steps_per_day = int(round(1/dt))
    if not np.isclose(steps_per_day * dt, 1.0):
        raise ValueError("DT must divide one day exactly.")

    # Arrays storing daily values
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    V = np.zeros(days)

    S[0] = S0
    I[0] = I0
    R[0] = R0
    V[0] = V0

    S_current = float(S0)
    I_current = float(I0)
    R_current = float(R0)
    V_current = float(V0)

    # Euler iteration
    for day in range(1, days):
        # dt may be smaller than one day
        for step in range(steps_per_day):

            # Rates from SIRV differential equations
            infection_rate = (beta * S_current * I_current / N)
            recovery_rate = (gamma * I_current)
            vaccination_rate = (nu * S_current)

            # Euler increments
            new_infections = (infection_rate * dt)
            new_recoveries = (recovery_rate * dt)
            new_vaccinations = (vaccination_rate * dt)

            # Avoid negative compartments cause by numerical approximation
            new_infections = min(max(new_infections, 0.0), S_current)
            new_vaccinations = min(max(new_vaccinations, 0.0), max(S_current - new_infections, 0.0))
            new_recoveries = min(max(new_recoveries, 0.0), I_current)

            # Forward Euler update
            S_next = (S_current - new_infections - new_vaccinations)
            I_next = (I_current + new_infections - new_recoveries)
            R_next = (R_current + new_recoveries)
            V_next = (V_current + new_vaccinations)

            # Move forward one Euler step
            S_current = S_next
            I_current = I_next
            R_current = R_next
            V_current = V_next

        # Save result once per day
        S[day] = S_current
        I[day] = I_current
        R[day] = R_current
        V[day] = V_next

    return S, I, R, V

# Fit beta for one weekly window
def fit_beta_for_window(S0, I0, R0, V0, nu, actual_smooth_new_cases):
    actual_smooth_new_cases = np.clip(actual_smooth_new_cases, 0, None)

    def loss(beta):
        S, I, R, V = run_sirv_euler(beta, GAMMA, nu, N, S0, I0, R0, V0, WINDOW_LEN)
        # daily infection incidence
        predicted_new_cases = (beta * I * S / N)
        log_predicted = np.log(predicted_new_cases + 1)
        log_actual = np.log(actual_smooth_new_cases + 1)

        return np.sum((log_predicted - log_actual)**2)

    result = minimize_scalar(loss, bounds=(BETA_MIN, BETA_MAX), method='bounded')

    return result.x

# Main rolling-window SIRV model
def main():
    confirmed_data = load_confirmed_data()
    vaccination_data = load_vaccination_data()
    simulation_start = pd.Timestamp("2020-03-01")
    simulation_end = pd.Timestamp("2023-03-09")

    # fit beta and nu for each week's historical data
    window_start_dates = []
    window_betas = []
    window_nus = []

    current = simulation_start

    while (current + pd.Timedelta(days=WINDOW_LEN) <= simulation_end):
        S0, I0, R0, V0 = (estimate_S_I_R_V(confirmed_data, vaccination_data, current))
        window = confirmed_data[(confirmed_data["date"] >= current) & (
            confirmed_data["date"] < current + pd.Timedelta(days=WINDOW_LEN)
        )]

        if len(window) < WINDOW_LEN:
            current += pd.Timedelta(days=WINDOW_LEN)
            continue

        # Estimate vaccination rate
        nu = estimate_nu_for_window(vaccination_data, current)

        # Fit beta
        beta = fit_beta_for_window(S0, I0, R0, V0, nu, window["new_cases_smooth"].values)

        window_start_dates.append(current)
        window_betas.append(beta)
        window_nus.append(nu)
        current += pd.Timedelta(days=WINDOW_LEN)


    print(f"Calculated {len(window_betas)}"
          f"weekly beta values.")

    # Predict week i using beta and nu from week i - 1
    all_dates = []
    all_predicted_new_cases = []
    all_actual_new_cases = []

    for i in range(1, len(window_betas)):
        this_week_start = (window_start_dates[i])
        # Only previous week's parameter
        beta_from_last_week = (window_betas[i-1])

        nu_from_last_week = window_nus[i-1]

        # Initial state known at beginning of prediction week
        S0, I0, R0, V0 = estimate_S_I_R_V(confirmed_data, vaccination_data, this_week_start)

        window = confirmed_data[(confirmed_data["date"] >= this_week_start) & (confirmed_data["date"] < this_week_start + pd.Timedelta(days=WINDOW_LEN))]

        if len(window) < WINDOW_LEN:
            continue

        # Euler SIRV prediction
        S, I, R, V = run_sirv_euler(beta_from_last_week, GAMMA, nu_from_last_week, N, S0, I0, R0, V0, WINDOW_LEN)
        predicted_new_cases = (beta_from_last_week * I * S / N)
        dates_in_window = pd.date_range(this_week_start, periods=WINDOW_LEN)

        all_dates.extend(dates_in_window)
        all_predicted_new_cases.extend(predicted_new_cases)
        all_actual_new_cases.extend(window["new_cases_raw"].values)

    # Error calculation
    all_dates = pd.to_datetime(all_dates)
    predicted = np.array(all_predicted_new_cases)
    actual = np.array(all_actual_new_cases)
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    print(f"SIRV Euler MAE: "
          f"{mae:.1f} cases/day")
    print(f"SIRV Euler RMSE: "
          f"{rmse:.1f} cases/day")

    # Plotting
    real_smoothed = (pd.Series(actual).rolling(7, center=True, min_periods=1).mean())

    plt.figure(figsize=(12, 8))
    plt.plot(all_dates, real_smoothed, label="Observed new cases (7-day average)")
    plt.plot(all_dates, predicted, label="SIRV Euler prediction")
    plt.title("SIRV Model Using Euler's Method Prediction")
    # plt.title("SIRV Model Using Forward Euler Method\n"
              # f"MAE: {mae:.0f}, "
              # f"RMSE: {rmse:.0f}, "
              # f"Euler step = {DT} day")
    plt.xlabel("Date")
    plt.ylabel("Daily New Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig("SIRV_Euler_prediction.png", dpi=150)
    plt.show()

# Run
if __name__ == "__main__":
    main()