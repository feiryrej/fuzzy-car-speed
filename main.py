import numpy as np
import matplotlib.pyplot as plt

class FuzzyLogicSystem:
    def __init__(self):
        # Define universe of discourse
        self.temp_range = np.arange(30, 90, 1)  # Temperature from 30°F to 89°F
        self.cloud_range = np.arange(0, 101, 1)  # Cloud cover from 0% to 100%
        self.speed_range = np.arange(0, 101, 1)  # Speed from 0 mph to 100 mph
        
    def trapmf(self, x, params):
        """Trapezoidal membership function - CORRECTED VERSION"""
        a, b, c, d = params
        
        # Ensure x is a numpy array for vectorized operations
        x = np.asarray(x)
        
        # Initialize output with zeros
        y = np.zeros_like(x, dtype=float)
        
        # Calculate membership values
        # Left slope: from a to b
        left_mask = (x >= a) & (x <= b)
        if b != a:
            y[left_mask] = (x[left_mask] - a) / (b - a)
        else:
            y[left_mask] = 1.0
        
        # Flat top: from b to c
        flat_mask = (x > b) & (x < c)
        y[flat_mask] = 1.0
        
        # Right slope: from c to d
        right_mask = (x >= c) & (x <= d)
        if d != c:
            y[right_mask] = (d - x[right_mask]) / (d - c)
        else:
            y[right_mask] = 1.0
        
        # Handle special case where b == c (triangular)
        if b == c:
            peak_mask = x == b
            y[peak_mask] = 1.0
        
        return np.clip(y, 0, 1)
    
    def trimf(self, x, params):
        """Triangular membership function - CORRECTED VERSION"""
        a, b, c = params
        
        # Ensure x is a numpy array for vectorized operations
        x = np.asarray(x)
        
        # Initialize output with zeros
        y = np.zeros_like(x, dtype=float)
        
        # Left slope: from a to b
        left_mask = (x >= a) & (x <= b)
        if b != a:
            y[left_mask] = (x[left_mask] - a) / (b - a)
        else:
            y[left_mask] = 1.0
        
        # Right slope: from b to c
        right_mask = (x > b) & (x <= c)
        if c != b:
            y[right_mask] = (c - x[right_mask]) / (c - b)
        else:
            y[right_mask] = 1.0
        
        # Peak point
        peak_mask = x == b
        y[peak_mask] = 1.0
        
        return np.clip(y, 0, 1)
    
    def gaussmf(self, x, params):
        """Gaussian membership function"""
        sigma, center = params
        x = np.asarray(x)
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def get_membership_value(self, x, func_type, params):
        """Get membership value for a single input or array"""
        if func_type == 'trapmf':
            return self.trapmf(x, params)
        elif func_type == 'trimf':
            return self.trimf(x, params)
        elif func_type == 'gaussmf':
            return self.gaussmf(x, params)
        else:
            raise ValueError("Unknown membership function type")
    
    def define_membership_functions(self):
        """Define membership functions for all fuzzy sets"""
        
        # Temperature membership functions
        self.temp_functions = {
            'Cool': ('trapmf', [30, 35, 55, 65]),    # Cool: peaks at 35-55°F, fades 30-65°F
            'Warm': ('trapmf', [55, 65, 85, 90])     # Warm: peaks at 65-85°F, fades 55-90°F
        }
        
        # Cloud Cover membership functions
        self.cloud_functions = {
            'Sunny': ('trapmf', [0, 5, 25, 35]),     # Sunny: peaks at 5-25%, fades 0-35%
            'Cloudy': ('trapmf', [25, 50, 90, 100])  # Cloudy: peaks at 50-90%, fades 25-100%
        }
        
        # Speed membership functions (for output)
        self.speed_functions = {
            'Slow': ('trapmf', [0, 10, 35, 45]),     # Slow: peaks at 10-35 mph, fades 0-45 mph
            'Fast': ('trapmf', [45, 60, 90, 100])    # Fast: peaks at 60-90 mph, fades 45-100 mph
        }
    
    def fuzzify_inputs(self, temperature, cloud_cover):
        """Fuzzify the crisp inputs"""
        # Temperature memberships
        cool_membership = self.get_membership_value(temperature, 
                                                   self.temp_functions['Cool'][0], 
                                                   self.temp_functions['Cool'][1])
        warm_membership = self.get_membership_value(temperature, 
                                                   self.temp_functions['Warm'][0], 
                                                   self.temp_functions['Warm'][1])
        
        # Cloud cover memberships
        sunny_membership = self.get_membership_value(cloud_cover, 
                                                    self.cloud_functions['Sunny'][0], 
                                                    self.cloud_functions['Sunny'][1])
        cloudy_membership = self.get_membership_value(cloud_cover, 
                                                     self.cloud_functions['Cloudy'][0], 
                                                     self.cloud_functions['Cloudy'][1])
        
        return {
            'Cool': cool_membership,
            'Warm': warm_membership,
            'Sunny': sunny_membership,
            'Cloudy': cloudy_membership
        }
    
    def apply_rules(self, memberships):
        """Apply fuzzy rules using min for AND operation - ORIGINAL 2 RULES ONLY"""
        # Rule 1: If Sunny AND Warm, then Fast
        rule1_strength = min(memberships['Sunny'], memberships['Warm'])
        
        # Rule 2: If Cloudy AND Cool, then Slow
        rule2_strength = min(memberships['Cloudy'], memberships['Cool'])
        
        return {
            'Fast': rule1_strength,
            'Slow': rule2_strength
        }
    
    def aggregate_outputs(self, rule_outputs):
        """Aggregate the rule outputs to create output fuzzy sets"""
        # Create output membership functions based on rule strengths
        fast_output = np.minimum(rule_outputs['Fast'], 
                                self.get_membership_value(self.speed_range, 
                                                         self.speed_functions['Fast'][0], 
                                                         self.speed_functions['Fast'][1]))
        
        slow_output = np.minimum(rule_outputs['Slow'], 
                                self.get_membership_value(self.speed_range, 
                                                         self.speed_functions['Slow'][0], 
                                                         self.speed_functions['Slow'][1]))
        
        # Combine using max (union)
        combined_output = np.maximum(fast_output, slow_output)
        
        return combined_output, {'Fast': fast_output, 'Slow': slow_output}
    
    def defuzzify(self, output_fuzzy_set):
        """Defuzzify using centroid method"""
        if np.sum(output_fuzzy_set) == 0:
            return 50  # Default moderate speed if no rules fire
        
        # Centroid defuzzification
        centroid = np.sum(self.speed_range * output_fuzzy_set) / np.sum(output_fuzzy_set)
        return centroid
    
    def evaluate(self, temperature, cloud_cover, show_details=False):
        """Main evaluation function"""
        # Step 1: Fuzzify inputs
        memberships = self.fuzzify_inputs(temperature, cloud_cover)
        
        # Step 2: Apply rules
        rule_outputs = self.apply_rules(memberships)
        
        # Step 3: Aggregate outputs
        combined_output, individual_outputs = self.aggregate_outputs(rule_outputs)
        
        # Step 4: Defuzzify
        final_speed = self.defuzzify(combined_output)
        
        if show_details:
            print(f"\n=== Fuzzy Logic Evaluation ===")
            print(f"Input - Temperature: {temperature}°F, Cloud Cover: {cloud_cover}%")
            print(f"\nFuzzification:")
            print(f"  Cool: {memberships['Cool']:.3f}")
            print(f"  Warm: {memberships['Warm']:.3f}")
            print(f"  Sunny: {memberships['Sunny']:.3f}")
            print(f"  Cloudy: {memberships['Cloudy']:.3f}")
            print(f"\nRule Evaluation:")
            print(f"  Rule 1 (Sunny AND Warm → Fast): {rule_outputs['Fast']:.3f}")
            print(f"  Rule 2 (Cloudy AND Cool → Slow): {rule_outputs['Slow']:.3f}")
            print(f"\nFinal Speed: {final_speed:.2f} mph")
        
        return final_speed, memberships, rule_outputs
    
    def plot_membership_functions(self):
        """Plot all membership functions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature membership functions
        axes[0, 0].plot(self.temp_range, 
                       self.get_membership_value(self.temp_range, 
                                               self.temp_functions['Cool'][0], 
                                               self.temp_functions['Cool'][1]), 
                       'b-', label='Cool', linewidth=2)
        axes[0, 0].plot(self.temp_range, 
                       self.get_membership_value(self.temp_range, 
                                               self.temp_functions['Warm'][0], 
                                               self.temp_functions['Warm'][1]), 
                       'r-', label='Warm', linewidth=2)
        axes[0, 0].set_title('Temperature Membership Functions')
        axes[0, 0].set_xlabel('Temperature (°F)')
        axes[0, 0].set_ylabel('Membership')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cloud cover membership functions
        axes[0, 1].plot(self.cloud_range, 
                       self.get_membership_value(self.cloud_range, 
                                               self.cloud_functions['Sunny'][0], 
                                               self.cloud_functions['Sunny'][1]), 
                       'y-', label='Sunny', linewidth=2)
        axes[0, 1].plot(self.cloud_range, 
                       self.get_membership_value(self.cloud_range, 
                                               self.cloud_functions['Cloudy'][0], 
                                               self.cloud_functions['Cloudy'][1]), 
                       'g-', label='Cloudy', linewidth=2)
        axes[0, 1].set_title('Cloud Cover Membership Functions')
        axes[0, 1].set_xlabel('Cloud Cover (%)')
        axes[0, 1].set_ylabel('Membership')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Speed membership functions
        axes[1, 0].plot(self.speed_range, 
                       self.get_membership_value(self.speed_range, 
                                               self.speed_functions['Slow'][0], 
                                               self.speed_functions['Slow'][1]), 
                       'purple', label='Slow', linewidth=2)
        axes[1, 0].plot(self.speed_range, 
                       self.get_membership_value(self.speed_range, 
                                               self.speed_functions['Fast'][0], 
                                               self.speed_functions['Fast'][1]), 
                       'orange', label='Fast', linewidth=2)
        axes[1, 0].set_title('Speed Membership Functions')
        axes[1, 0].set_xlabel('Speed (mph)')
        axes[1, 0].set_ylabel('Membership')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Remove the unused subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.show()

def test_fuzzy_system():
    """Test the fuzzy system with the provided test cases"""
    fuzzy_system = FuzzyLogicSystem()
    fuzzy_system.define_membership_functions()
    
    # Test cases from the lab
    test_cases = [
        (65, 25),  # Temp = 65°F, Cloud Cover = 25%
        (62, 47),  # Temp = 62°F, Cloud Cover = 47%
        (75, 30),  # Temp = 75°F, Cloud Cover = 30%
        (53, 65),  # Temp = 53°F, Cloud Cover = 65%
        (68, 70)   # Temp = 68°F, Cloud Cover = 70%
    ]
    
    print("=== FUZZY LOGIC DRIVING SPEED SYSTEM ===\n")
    
    for i, (temp, cloud) in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        speed, memberships, rules = fuzzy_system.evaluate(temp, cloud, show_details=True)
        print("-" * 50)
    
    return fuzzy_system

# Run the test
if __name__ == "__main__":
    system = test_fuzzy_system()
    
    # Uncomment the line below to see the membership function plots
    # system.plot_membership_functions()
    
    # Interactive testing
    print("\n=== INTERACTIVE TESTING ===")
    print("You can now test with custom values:")
    
    try:
        temp = float(input("Enter temperature (°F): "))
        cloud = float(input("Enter cloud cover (%): "))
        speed, _, _ = system.evaluate(temp, cloud, show_details=True)
        print(f"\nRecommended driving speed: {speed:.2f} mph")
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")