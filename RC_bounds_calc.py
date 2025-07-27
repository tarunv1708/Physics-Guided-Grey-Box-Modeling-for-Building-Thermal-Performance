import pandas as pd

# Constants
A_roof = 921.35  # m²
variation = {
    "R_membrane": 0.4,
    "R_insulation": 0.35,
    "R_metal_deck": 0.2,
    #"R_ceiling_insulation": 0.3,
    #"R_metal_studs": 0.5,
    #"R_gypsum": 0.3,
    "C_membrane": 0.3,
    "C_insulation": 0.35,
    "C_metal_deck": 0.2,
    #"C_ceiling_insulation": 0.3,
    #"C_metal_studs": 0.5,
    #"C_gypsum": 0.3,
    "C_air": 0.25
}

# Material properties
material_props = {
    "roof_membrane": {"k": 0.16, "density": 950, "cp": 1800, "thickness": 0.003},
    "insulation": {"k": 0.035, "density": 40, "cp": 1400, "thickness": 0.127},
    "metal_deck": {"k": 50, "density": 7850, "cp": 500, "thickness": 0.0127},
    #"ceiling_insulation": {"k": 0.035, "density": 40, "cp": 1400, "thickness": 0.0254},
    #"metal_studs": {"k": 50, "density": 7850, "cp": 500, "thickness": 0.002},
    #"gypsum": {"k": 0.16, "density": 785, "cp": 830, "thickness": 0.0159},
    "air": {"density": 1.225, "cp": 1005, "volume": 3.0}  # volume in m³
}

# Function to compute resistance and capacitance
def compute_sobol_bounds(material_props, A_roof, variation):
    def R(material): return material["thickness"] / (material["k"] * A_roof)
    def C(material): return material["density"] * material["cp"] * material["thickness"] * A_roof

    R_values = {
        "R_membrane": R(material_props["roof_membrane"]),
        "R_insulation": R(material_props["insulation"]),
        "R_metal_deck": R(material_props["metal_deck"]),
        #"R_ceiling_insulation": R(material_props["ceiling_insulation"]),
        #"R_metal_studs": R(material_props["metal_studs"]),
        #"R_gypsum": R(material_props["gypsum"])
    }

    C_values = {
        "C_membrane": C(material_props["roof_membrane"]),
        "C_insulation": C(material_props["insulation"]),
        "C_metal_deck": C(material_props["metal_deck"]),
        #"C_ceiling_insulation": C(material_props["ceiling_insulation"]),
        #"C_metal_studs": C(material_props["metal_studs"]),
       # "C_gypsum": C(material_props["gypsum"]),
        "C_air": material_props["air"]["density"] * material_props["air"]["cp"] * material_props["air"]["volume"] * A_roof
    }

    bounds = []
    records = []

    for key in list(R_values.keys()) + list(C_values.keys()):
        nominal = R_values.get(key) or C_values.get(key)
        perc = variation[key]
        lower = nominal * (1 - perc)
        upper = nominal * (1 + perc)
        bounds.append((lower, upper))
        records.append({"Parameter": key, "Nominal": nominal, "Lower": lower, "Upper": upper})

    bounds_df = pd.DataFrame(records).set_index("Parameter")
    return bounds, bounds_df

# Run and display
bounds, bounds_df = compute_sobol_bounds(material_props, A_roof, variation)

# Print nicely
print("\n🔍 Sobol Parameter Bounds:\n")
print(bounds_df)

# Save to CSV
bounds_df.to_csv("RC_Sobol_Bounds_Verification.csv")
print("\n✅ Saved to RC_Sobol_Bounds_Verification.csv")
