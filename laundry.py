import matplotlib.pyplot as plt

# --- Membership Function Definitions ---
MFS_DEFINITION = {
    "dirtiness": {
        "Low": [(0, 1), (2, 1), (4, 0), (10, 0)],
        "Medium": [(0, 0), (2, 0), (5, 1), (8, 0), (10, 0)],
        "High": [(0, 0), (6, 0), (8, 1), (10, 1)],
    },
    "quantity": {
        "Light": [(0, 1), (2, 1), (5, 0), (12, 0)],
        "Medium": [(0, 0), (3, 0), (6, 1), (9, 0), (12, 0)],
        "Heavy": [(0, 0), (7, 0), (10, 1), (12, 1)],
    },
    "intensity": {
        "Light": [(0, 1), (20, 1), (40, 0), (100, 0)],
        "Normal": [(0, 0), (30, 0), (50, 1), (70, 0), (100, 0)],
        "Strong": [(0, 0), (60, 0), (80, 1), (100, 1)],
    }
}

# --- Membership Function Calculation ---
def get_membership(input_val, points):
    if not points:
        return 0.0

    if input_val <= points[0][0]:
        return points[0][1]
    if input_val >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x1 <= input_val <= x2:
            if input_val == x1: return y1
            if input_val == x2: return y2
            if y1 == y2: return y1
            if x1 == x2: return y1 

            # Linear interpolation
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return slope * input_val + intercept

    return 0.0


# --- Fuzzification ---
def fuzzify(input_val, mfs):
    memberships = {}
    for set_name, points in mfs.items():
        memberships[set_name] = get_membership(input_val, points)
    return memberships


# --- Rule Evaluation ---
def apply_rules(dirtiness_mfs, quantity_mfs):
    intensity_activations = {"Light": 0.0, "Normal": 0.0, "Strong": 0.0}

    # Rule 1: If D(Low) and Q(Light) => Light
    rule1 = min(dirtiness_mfs.get("Low", 0.0), quantity_mfs.get("Light", 0.0))
    intensity_activations["Light"] = max(intensity_activations["Light"], rule1)

    # Rule 2: If D(Low) and Q(Medium) => Light
    rule2 = min(dirtiness_mfs.get("Low", 0.0), quantity_mfs.get("Medium", 0.0))
    intensity_activations["Light"] = max(intensity_activations["Light"], rule2)

    # Rule 3: If D(Low) and Q(Heavy) => Strong
    rule3 = min(dirtiness_mfs.get("Low", 0.0), quantity_mfs.get("Heavy", 0.0))
    intensity_activations["Strong"] = max(intensity_activations["Strong"], rule3)

    # Rule 4: If D(Medium) and Q(Light) => Light
    rule4 = min(dirtiness_mfs.get("Medium", 0.0), quantity_mfs.get("Light", 0.0))
    intensity_activations["Light"] = max(intensity_activations["Light"], rule4)

    # Rule 5: If D(Medium) and Q(Medium) => Normal
    rule5 = min(dirtiness_mfs.get("Medium", 0.0), quantity_mfs.get("Medium", 0.0))
    intensity_activations["Normal"] = max(intensity_activations["Normal"], rule5)

    # Rule 6: If D(Medium) and Q(Heavy) => Strong
    rule6 = min(dirtiness_mfs.get("Medium", 0.0), quantity_mfs.get("Heavy", 0.0))
    intensity_activations["Strong"] = max(intensity_activations["Strong"], rule6)

    # Rule 7: If D(High) and Q(Light) => Light
    rule7 = min(dirtiness_mfs.get("High", 0.0), quantity_mfs.get("Light", 0.0))
    intensity_activations["Light"] = max(intensity_activations["Light"], rule7)

    # Rule 8: If D(High) and Q(Medium) => Strong
    rule8 = min(dirtiness_mfs.get("High", 0.0), quantity_mfs.get("Medium", 0.0))
    intensity_activations["Strong"] = max(intensity_activations["Strong"], rule8)

    # Rule 9: If D(High) and Q(Heavy) => Strong
    rule9 = min(dirtiness_mfs.get("High", 0.0), quantity_mfs.get("Heavy", 0.0))
    intensity_activations["Strong"] = max(intensity_activations["Strong"], rule9)

    return intensity_activations


# --- Output Aggregation ---
def aggregate(x_intensity, activations, intensity_mfs):
    agg_value = 0.0
    for set_name, act_strength in activations.items():
        if act_strength > 0:
            original = get_membership(x_intensity, intensity_mfs[set_name])
            clipped = min(act_strength, original)
            agg_value = max(agg_value, clipped)
    return agg_value


# --- Defuzzification (COG) ---
def defuzzify(activations, intensity_mfs, num_samples=101):
    min_x, max_x = 0, 100
    x_samples = [min_x + i * (max_x - min_x) / (num_samples - 1) for i in range(num_samples)]

    num_sum = 0.0
    denom_sum = 0.0
    agg_points = []

    for x in x_samples:
        y = aggregate(x, activations, intensity_mfs)
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
        ax.plot(x_range, y_vals, label=mf_name, linewidth=2)

    if input_val is not None and fuz_vals is not None:
        ax.vlines(input_val, 0, 1, colors='r', linestyles='dashed', label=f"Input = {input_val:.2f}", linewidth=2)
        for mf_name, mem_deg in fuz_vals.items():
            if mem_deg > 0.001:
                ax.hlines(mem_deg, min_x, input_val, colors='gray', linestyles='dotted', alpha=0.7)
                ax.plot(input_val, mem_deg, 'ro', markersize=8)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min_x, max_x)


def plot_agg(ax, agg_pts, cog, activations, intensity_mfs):
    ax.set_title("Aggregated Output and Defuzzification")
    ax.set_xlabel("Wash Intensity")
    ax.set_ylabel("Membership Degree")
    ax.grid(True, linestyle='--', alpha=0.7)

    x_range_int = [i * 0.5 for i in range(201)]

    # Plot original membership functions
    for mf_name, pts in intensity_mfs.items():
        y_vals = [get_membership(x, pts) for x in x_range_int]
        ax.plot(x_range_int, y_vals, label=f"{mf_name}", linestyle='dashed', alpha=0.7, linewidth=2)

    # Plot clipped membership functions
    for mf_name, act_strength in activations.items():
        if act_strength > 0:
            clipped = [min(act_strength, get_membership(x, intensity_mfs[mf_name])) for x in x_range_int]
            ax.plot(x_range_int, clipped, linestyle='--', label=f"Clipped '{mf_name}'", alpha=0.8, linewidth=2)

    # Plot aggregated output
    x_agg = [p[0] for p in agg_pts]
    y_agg = [p[1] for p in agg_pts]
    ax.plot(x_agg, y_agg, color='purple', linewidth=3, label="Aggregated Output Set")
    ax.fill_between(x_agg, y_agg, color='purple', alpha=0.3)

    # Plot center of gravity
    ax.vlines(cog, 0, max(y_agg + [1.0]), colors='r', linestyles='solid', linewidth=3, label=f"COG = {cog:.2f}")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 100)


# --- Main Program Loop ---
if __name__ == "__main__":
    print("=== WASHING MACHINE FUZZY LOGIC CONTROL SYSTEM ===")
    print("This system determines wash intensity based on:")
    print("- Dirtiness Level (0-10 scale)")
    print("- Laundry Quantity (0-12 kg)")
    print("Output: Wash Intensity (0-100 scale)")
    print()
    
    while True:
        # --- Input Section ---
        while True:
            try:
                dirtiness = float(input("Enter dirtiness level (0-10): "))
                if 0 <= dirtiness <= 10:
                    break
                else:
                    print("Please enter a value between 0 and 10.")
            except ValueError:
                print("Invalid dirtiness input. Try again.")

        while True:
            try:
                quantity = float(input("Enter laundry quantity in kg (0-12): "))
                if 0 <= quantity <= 12:
                    break
                else:
                    print("Please enter a value between 0 and 12 kg.")
            except ValueError:
                print("Invalid quantity input. Try again.")

        print(f"\n=== PROCESSING: Dirtiness={dirtiness}, Quantity={quantity}kg ===")

        # --- Fuzzification ---
        dirtiness_mfs = fuzzify(dirtiness, MFS_DEFINITION["dirtiness"])
        quantity_mfs = fuzzify(quantity, MFS_DEFINITION["quantity"])

        print("\n--- Dirtiness Level Fuzzification ---")
        for k, v in dirtiness_mfs.items():
            print(f"{k}: {v:.4f}")

        print("\n--- Laundry Quantity Fuzzification ---")
        for k, v in quantity_mfs.items():
            print(f"{k}: {v:.4f}")

        # --- Rule Evaluation ---
        intensity_acts = apply_rules(dirtiness_mfs, quantity_mfs)
        print("\n--- Rule-Based Intensity Activations ---")
        for k, v in intensity_acts.items():
            print(f"{k}: {v:.4f}")

        # --- Defuzzification ---
        intensity_cog, agg_curve = defuzzify(intensity_acts, MFS_DEFINITION["intensity"])
        print(f"\nDefuzzified Wash Intensity Output (COG): {intensity_cog:.3f}")

        # --- Interpret Result ---
        if intensity_cog < 30:
            wash_type = "Light Wash"
        elif intensity_cog < 65:
            wash_type = "Normal Wash"
        else:
            wash_type = "Strong Wash"
        
        print(f"Recommended Wash Type: {wash_type}")

        # --- Defuzz Table Display ---
        print("\n" + "="*50)
        print("DEFUZZIFICATION TABLE")
        print("="*50)
        print("|   X   |   Y   |   X * Y     |")
        print("-"*35)
        sum_y, sum_xy = 0.0, 0.0
        for x in range(0, 101, 5):
            y = aggregate(x, intensity_acts, MFS_DEFINITION["intensity"])
            xy = x * y
            sum_y += y
            sum_xy += xy
            print(f"{x:6} {y:7.4f} {xy:12.4f}")
        print("-"*35)
        print(f"Sum Y: {sum_y:.4f}, Sum XY: {sum_xy:.4f}")
        if sum_y > 0:
            print(f"COG = {sum_xy:.4f} / {sum_y:.4f} = {sum_xy/sum_y:.5f}")
        else:
            print("COG = 0 (no activation)")

        # --- Plotting ---
        fig, axs = plt.subplots(3, 1, figsize=(14, 20))
        plt.subplots_adjust(hspace=0.4, right=0.75)
        
        plot_mfs(axs[0], "Dirtiness Level", MFS_DEFINITION["dirtiness"], dirtiness, dirtiness_mfs)
        plot_mfs(axs[1], "Laundry Quantity (kg)", MFS_DEFINITION["quantity"], quantity, quantity_mfs)
        plot_agg(axs[2], agg_curve, intensity_cog, intensity_acts, MFS_DEFINITION["intensity"])
        
        plt.suptitle("Washing Machine Fuzzy Logic Control System", fontsize=16, y=0.97)
        plt.show()

        # --- Ask User to Run Again ---
        repeat = input("\nWould you like to enter new laundry parameters? (y/n): ").strip().lower()
        if repeat != 'y':
            print("Exiting washing machine control system. Goodbye!")
            break