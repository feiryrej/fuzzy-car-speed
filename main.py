import matplotlib.pyplot as plt

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


# Membership Function Calculation
def get_membership(input, points):
    if not points: 
        return 0.0

    if input <= points[0][0]:
        return points[0][1]
    if input >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        x1, y1 = p1
        x2, y2 = p2

        if x1 <= input <= x2:
            if input == x1: return y1
            if input == x2: return y2
            
            if y1 == y2:
                return y1
            
            if x1 == x2: 
                return y1
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            membership = slope * input + intercept
            return membership
            
    return 0.0


# Fuzzification
def fuzzify(input, mfs):
    memberships = {}
    for set_name, points in mfs.items():
        memberships[set_name] = get_membership(input, points)
    return memberships


# Rule Definition and Evaluation
def apply_rules(temp_mfs, cover_mfs):
    speed_activations = {"Slow": 0.0, "Fast": 0.0} # Initialize output activations

    # Rule 1: IF Temp IS Warm AND Cover IS Sunny THEN Speed IS Fast
    warm_degree = temp_mfs.get("Warm", 0.0)
    sunny_degree = cover_mfs.get("Sunny", 0.0)
    rule1_strength = min(warm_degree, sunny_degree)
    speed_activations["Fast"] = max(speed_activations["Fast"], rule1_strength)

    # Rule 2: IF Temp IS Cool AND Cover IS Partly THEN Speed IS Slow
    cool_degree = temp_mfs.get("Cool", 0.0)
    partly_degree = cover_mfs.get("Partly", 0.0)
    rule2_strength = min(cool_degree, partly_degree)
    speed_activations["Slow"] = max(speed_activations["Slow"], rule2_strength)
    
    return speed_activations


# Aggregation of Output Fuzzy Sets
def aggregate(x_speed, activations, speed_mfs):
    agg_value = 0.0
    
    for set_name, act_strength in activations.items():
        if act_strength > 0:
            original_mfs = get_membership(x_speed, speed_mfs[set_name])
            clipped_mfs = min(act_strength, original_mfs)
            agg_value = max(agg_value, clipped_mfs)
            
    return agg_value


# Defuzzification
def defuzzify(activations, speed_mfs, num_samples=101):
    min_x = 0
    max_x = 100
    
    x_samples = [min_x + i * (max_x - min_x) / (num_samples - 1) for i in range(num_samples)]
    
    numerator_sum = 0.0
    denominator_sum = 0.0
    
    agg_points = []

    for x in x_samples:
        y = aggregate(x, activations, speed_mfs)
        numerator_sum += x * y
        denominator_sum += y
        agg_points.append((x, y))
        
    if denominator_sum == 0:
        return 0.0, agg_points 
        
    cog = numerator_sum / denominator_sum
    return cog, agg_points


# Plotting Functions
def plot_mfs(ax, var_name, mfs_data, input_val=None, fuz_vals=None):
    ax.set_title(f"Membership Functions for {var_name}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Determine plot range from data
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

    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min_x, max_x)


def plot_agg(ax, agg_pts, cog, activations, speed_mfs):
    ax.set_title("Aggregated Output and Defuzzification")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle='--', alpha=0.7)

    min_x_spd = 0
    max_x_spd = 100
    x_range_spd = [min_x_spd + i * (max_x_spd - min_x_spd) / 200 for i in range(201)]

    for mf_name, pts in speed_mfs.items():
        y_vals = [get_membership(x, pts) for x in x_range_spd]
        ax.plot(x_range_spd, y_vals, label=f"{mf_name}", linestyle='dashed', alpha=0.7)

    for mf_name, act_strength in activations.items():
        if act_strength > 0:
            orig_mf_pts = speed_mfs[mf_name]
            clip_y_vals = [min(act_strength, get_membership(x, orig_mf_pts)) for x in x_range_spd]
            ax.plot(x_range_spd, clip_y_vals, linestyle='--', label=f"Clipped '{mf_name}' (act: {act_strength:.2f})", alpha=0.8)

    x_agg = [p[0] for p in agg_pts]
    y_agg = [p[1] for p in agg_pts]
    
    sorted_agg_pts = sorted(agg_pts, key=lambda p: p[0])
    x_agg_sorted = [p[0] for p in sorted_agg_pts]
    y_agg_sorted = [p[1] for p in sorted_agg_pts]
    
    ax.plot(x_agg_sorted, y_agg_sorted, label="Aggregated Output Set", color='purple', linewidth=2)
    ax.fill_between(x_agg_sorted, y_agg_sorted, color='purple', alpha=0.3)
    
    max_y_plot = 1.0
    if y_agg_sorted:
        max_y_plot = max(max_y_plot, max(y_agg_sorted))

    ax.vlines(cog, 0, max_y_plot, colors='r', linestyles='solid', linewidth=2, label=f"COG (Output Speed) = {cog:.2f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min_x_spd, max_x_spd)


if __name__ == "__main__":
    while True:
        try:
            input_temp = input("Enter temperature in deg F: ")
            temp = float(input_temp)
            break
        except ValueError:
            print("Invalid input.")

    while True:
        try:
            input_cover = input("Enter cloud cover %: ")
            cover = float(input_cover)
            break 
        except ValueError:
            print("Invalid input.")

    # --- Step 1: Fuzzification ---
    temp_mfs = fuzzify(temp, MFS_DEFINITION["temperature"])
    print("Temperature Fuzzification Results:")
    for set_name, membership in temp_mfs.items():
        print(f"  Membership in {set_name}: {membership:.3f}")
    
    cover_mfs = fuzzify(cover, MFS_DEFINITION["cover"])
    print("\nCloud Cover Fuzzification Results:")
    for set_name, membership in cover_mfs.items():
        print(f"  Membership in {set_name}: {membership:.3f}")

    # --- Step 2: Rule Evaluation ---
    speed_acts = apply_rules(temp_mfs, cover_mfs)
    print("\nSpeed Output Activations (from Rules):")
    for set_name, strength in speed_acts.items():
        print(f"  Activation for {set_name}: {strength:.3f}")

    # --- Step 3: Defuzzification ---
    n_samps_table = 21
    n_samps_calc = 101

    speed_cog, agg_curve = defuzzify(
        speed_acts,
        MFS_DEFINITION["speed"],
        num_samples=n_samps_calc
    )
    print(f"\n--- Defuzzified Output Speed (COG): {speed_cog:.3f} ---\n")
    print("------------------------------------------")
    print("|      X      |     Y     |     X * Y     |")
    print("------------------------------------------")
        
    x_col = [i * 5 for i in range(21)]
    sum_y = 0.0
    sum_xy = 0.0

    for x in x_col:
        y = aggregate(x, speed_acts, MFS_DEFINITION["speed"])
        xy = x * y
        sum_y += y
        sum_xy += xy
        print(f"{x:10.0f} {y:15.3f} {x:12.3f}")

    print("------------------------------------------")
    print(f"Total Sum: {sum_y:16.3f} {sum_xy:12.3f}")
    
    cog_from_table = sum_xy / sum_y
    print(f"\nCOG = SUM(xy) / SUM(y)")
    print(f"COG = {sum_xy:.3f} / {sum_y:.3f} = {cog_from_table:.5f}")

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.5, right=0.75)
    plot_mfs(axs[0], "Temperature", MFS_DEFINITION["temperature"], temp, temp_mfs)
    plot_mfs(axs[1], "Cloud Cover", MFS_DEFINITION["cover"], cover, cover_mfs)
    plot_agg(axs[2], agg_curve, speed_cog, speed_acts, MFS_DEFINITION["speed"])
    plt.suptitle("Fuzzy Logic System for Speed Control", fontsize=16, y=0.96)
    plt.show()

# Fuzzy Logic Lab 1
    # Temp = 65 deg F and Cloud Cover = 25%? 
    # Temp = 62 deg F and Cloud Cover = 47%?
    # Temp = 75 deg F and Cloud Cover = 30%?
    # Temp = 53 deg F and Cloud Cover = 65%?
    # Temp = 68 deg F and Cloud Cover = 70%?