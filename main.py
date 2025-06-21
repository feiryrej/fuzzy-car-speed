import matplotlib.pyplot as plt

# --- Membership Function Definitions ---
MFS_DEFINITION = {
    "temperature": {
        "Freezing": [(0, 1), (30, 1), (50, 0), (110, 0)],
        "Cool": [(0, 0), (30, 0), (50, 1), (70, 0), (110, 0)],
        "Warm": [(0, 0), (50, 0), (70, 1), (90, 0), (110, 0)],
        "Hot": [(0, 0), (70, 0), (90, 1), (110, 1)],
    },
    "cover": {
        "Sunny": [(0, 1), (20, 1), (40, 0), (100, 0)],
        "Partly": [(0, 0), (20, 0), (50, 1), (80, 0), (100, 0)],
        "Overcast": [(0, 0), (60, 0), (80, 1), (100, 1)],
    },
    "speed": {
        "Slow": [(0, 1), (25, 1), (75, 0), (100, 0)],
        "Fast": [(0, 0), (25, 0), (75, 1), (100, 1)],
    }
}

# --- Membership Function Calculation ---
def get_membership(input, points):
    if not points:
        return 0.0

    if input <= points[0][0]:
        return points[0][1]
    if input >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x1 <= input <= x2:
            if input == x1: return y1
            if input == x2: return y2
            if y1 == y2: return y1
            if x1 == x2: return y1 

            # Linear interpolation
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return slope * input + intercept

    return 0.0


# --- Fuzzification ---
def fuzzify(input, mfs):
    memberships = {}
    for set_name, points in mfs.items():
        memberships[set_name] = get_membership(input, points)
    return memberships


# --- Rule Evaluation ---
def apply_rules(temp_mfs, cover_mfs):
    speed_activations = {"Slow": 0.0, "Fast": 0.0}

    rule1 = min(temp_mfs.get("Warm", 0.0), cover_mfs.get("Sunny", 0.0))
    rule2 = min(temp_mfs.get("Cool", 0.0), cover_mfs.get("Partly", 0.0))

    speed_activations["Fast"] = max(speed_activations["Fast"], rule1)
    speed_activations["Slow"] = max(speed_activations["Slow"], rule2)

    return speed_activations


# --- Output Aggregation ---
def aggregate(x_speed, activations, speed_mfs):
    agg_value = 0.0
    for set_name, act_strength in activations.items():
        if act_strength > 0:
            original = get_membership(x_speed, speed_mfs[set_name])
            clipped = min(act_strength, original)
            agg_value = max(agg_value, clipped)
    return agg_value


# --- Defuzzification (COG) ---
def defuzzify(activations, speed_mfs, num_samples=101):
    min_x, max_x = 0, 100
    x_samples = [min_x + i * (max_x - min_x) / (num_samples - 1) for i in range(num_samples)]

    num_sum = 0.0
    denom_sum = 0.0
    agg_points = []

    for x in x_samples:
        y = aggregate(x, activations, speed_mfs)
        num_sum += x * y
        denom_sum += y
        agg_points.append((x, y))

    if denom_sum == 0:
        return 0.0, agg_points

    return num_sum / denom_sum, agg_points


# --- Plotting Functions ---
def plot_mfs(ax, var_name, mfs_data, input_val=None, fuz_vals=None):
    ax.set_title(f"Membership Functions for {var_name}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle='--', alpha=0.7)

    all_x = [p[0] for mf_d in mfs_data.values() for p in mf_d]
    min_x, max_x = min(all_x), max(all_x)
    x_range = [min_x + i * (max_x - min_x) / 200 for i in range(201)]

    for mf_name, pts in mfs_data.items():
        y_vals = [get_membership(x, pts) for x in x_range]
        ax.plot(x_range, y_vals, label=mf_name)

    if input_val is not None and fuz_vals is not None:
        ax.vlines(input_val, 0, 1, colors='r', linestyles='dashed', label=f"Input = {input_val:.2f}")
        for mf_name, mem_deg in fuz_vals.items():
            if mem_deg > 0.001:
                ax.hlines(mem_deg, min_x, input_val, colors='gray', linestyles='dotted', alpha=0.7)
                ax.plot(input_val, mem_deg, 'ro', markersize=5)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min_x, max_x)


def plot_agg(ax, agg_pts, cog, activations, speed_mfs):
    ax.set_title("Aggregated Output and Defuzzification")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle='--', alpha=0.7)

    x_range_spd = [i * 0.5 for i in range(201)]

    for mf_name, pts in speed_mfs.items():
        y_vals = [get_membership(x, pts) for x in x_range_spd]
        ax.plot(x_range_spd, y_vals, label=f"{mf_name}", linestyle='dashed', alpha=0.7)

    for mf_name, act_strength in activations.items():
        if act_strength > 0:
            clipped = [min(act_strength, get_membership(x, speed_mfs[mf_name])) for x in x_range_spd]
            ax.plot(x_range_spd, clipped, linestyle='--', label=f"Clipped '{mf_name}'", alpha=0.8)

    x_agg = [p[0] for p in agg_pts]
    y_agg = [p[1] for p in agg_pts]
    ax.plot(x_agg, y_agg, color='purple', linewidth=2, label="Aggregated Output Set")
    ax.fill_between(x_agg, y_agg, color='purple', alpha=0.3)

    ax.vlines(cog, 0, max(y_agg + [1.0]), colors='r', linestyles='solid', linewidth=2, label=f"COG = {cog:.2f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 100)


# --- Main Program Loop ---
if __name__ == "__main__":
    while True:
        # --- Input Section ---
        while True:
            try:
                temp = float(input("Enter temperature in °F: "))
                break
            except ValueError:
                print("Invalid temperature input. Try again.")

        while True:
            try:
                cover = float(input("Enter cloud cover %: "))
                break
            except ValueError:
                print("Invalid cloud cover input. Try again.")

        # --- Fuzzification ---
        temp_mfs = fuzzify(temp, MFS_DEFINITION["temperature"])
        cover_mfs = fuzzify(cover, MFS_DEFINITION["cover"])

        print("\n--- Temperature Fuzzification ---")
        for k, v in temp_mfs.items():
            print(f"{k}: {v:.3f}")

        print("\n--- Cloud Cover Fuzzification ---")
        for k, v in cover_mfs.items():
            print(f"{k}: {v:.3f}")

        # --- Rule Evaluation ---
        speed_acts = apply_rules(temp_mfs, cover_mfs)
        print("\n--- Rule-Based Speed Activations ---")
        for k, v in speed_acts.items():
            print(f"{k}: {v:.3f}")

        # --- Defuzzification ---
        speed_cog, agg_curve = defuzzify(speed_acts, MFS_DEFINITION["speed"])
        print(f"\nDefuzzified Speed Output (COG): {speed_cog:.3f}")

        # --- Defuzz Table Display ---
        print("\n-------------------------------")
        print("|   X   |   Y   |   X * Y     |")
        print("-------------------------------")
        sum_y, sum_xy = 0.0, 0.0
        for x in range(0, 101, 5):
            y = aggregate(x, speed_acts, MFS_DEFINITION["speed"])
            xy = x * y
            sum_y += y
            sum_xy += xy
            print(f"{x:6} {y:7.3f} {xy:12.3f}")
        print("-------------------------------")
        print(f"Sum Y: {sum_y:.3f}, Sum XY: {sum_xy:.3f}")
        print(f"COG = {sum_xy:.3f} / {sum_y:.3f} = {sum_xy/sum_y:.5f}")

        # --- Plotting ---
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
        plt.subplots_adjust(hspace=0.5, right=0.75)
        plot_mfs(axs[0], "Temperature", MFS_DEFINITION["temperature"], temp, temp_mfs)
        plot_mfs(axs[1], "Cloud Cover", MFS_DEFINITION["cover"], cover, cover_mfs)
        plot_agg(axs[2], agg_curve, speed_cog, speed_acts, MFS_DEFINITION["speed"])
        plt.suptitle("Fuzzy Logic Speed Decision System", fontsize=16, y=0.96)
        plt.show()

        # --- Ask User to Run Again ---
        repeat = input("\nWould you like to enter new values? (y/n): ").strip().lower()
        if repeat != 'y':
            print("Exiting program. Goodbye!")
            break
