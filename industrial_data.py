# ✅ Custom labeled industrial maintenance reports dataset
# 1 = Positive (normal/good), 0 = Negative (issue/urgent)

industrial_reports = [
    ("The motor is running smoothly with no unusual vibrations detected.", 1),
    ("Bearing temperature has exceeded safe limits and requires immediate attention.", 0),
    ("Routine inspection completed, all systems operating within normal parameters.", 1),
    ("Critical failure detected in the cooling system, production halted.", 0),
    ("The robotic arm calibration was successful and performance has improved significantly.", 1),
    ("Insulation damage found on multiple cables, urgent replacement needed.", 0),
    ("All sensors are functioning correctly with optimal readings.", 1),
    ("Pressure valve malfunction caused unexpected shutdown.", 0),
    ("Maintenance check passed with no issues found.", 1),
    ("Severe oil leakage detected near the hydraulic system.", 0),
    ("System upgrade completed, performance metrics show improvement.", 1),
    ("Overheating reported in control panel, immediate inspection required.", 0),
    ("Conveyor belt operating efficiently with consistent speed.", 1),
    ("Multiple error codes triggered, system requires diagnostic review.", 0),
    ("Quality control passed, all units meet specification.", 1),
    ("Unusual noise detected from gearbox, recommend inspection.", 0),
    ("Calibration successful, equipment ready for production.", 1),
    ("Power supply fluctuation causing intermittent system failures.", 0),
    ("Preventive maintenance completed ahead of schedule.", 1),
    ("Safety sensor malfunction, line stopped for inspection.", 0),
    ("New firmware update improved response time significantly.", 1),
    ("Corrosion detected on metal components, replacement scheduled.", 0),
    ("Production line running at optimal efficiency today.", 1),
    ("Emergency stop triggered due to obstruction in pathway.", 0),
    ("All safety protocols verified and functioning correctly.", 1),
    ("Wiring fault identified in control circuit, repair needed.", 0),
    ("Equipment performance exceeds expected benchmarks.", 1),
    ("Hydraulic pressure dropped below minimum threshold.", 0),
    ("Successful test run completed with no anomalies.", 1),
    ("Critical alarm triggered for temperature exceeding limits.", 0),
]

print(f"✅ Total labeled samples: {len(industrial_reports)}")
print(f"Positive samples: {sum(1 for _, label in industrial_reports if label == 1)}")
print(f"Negative samples: {sum(1 for _, label in industrial_reports if label == 0)}")