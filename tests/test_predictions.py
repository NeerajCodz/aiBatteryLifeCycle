import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging; logging.disable(logging.WARNING)

from api.model_registry import registry
registry.load_all()

logging.disable(logging.NOTSET)

tests = [
    ('Early life cycle 5',   {'cycle_number':5,'ambient_temperature':24,'peak_voltage':4.19,'min_voltage':2.61,'voltage_range':1.58,'avg_current':1.82,'avg_temp':32.6,'temp_rise':14.7,'cycle_duration':3690,'Re':0.045,'Rct':0.069,'delta_capacity':0.0}),
    ('Healthy cycle 100',    {'cycle_number':100,'ambient_temperature':24,'peak_voltage':4.19,'min_voltage':2.61,'voltage_range':1.58,'avg_current':1.82,'avg_temp':32.6,'temp_rise':14.7,'cycle_duration':3690,'Re':0.055,'Rct':0.09,'delta_capacity':-0.005}),
    ('Degraded cycle 160',   {'cycle_number':160,'ambient_temperature':40,'peak_voltage':4.1,'min_voltage':2.5,'voltage_range':1.6,'avg_current':2.5,'avg_temp':50,'temp_rise':25,'cycle_duration':3200,'Re':0.12,'Rct':0.18,'delta_capacity':-0.05}),
    ('EOL cycle 196',        {'cycle_number':196,'ambient_temperature':24,'peak_voltage':4.05,'min_voltage':2.2,'voltage_range':1.85,'avg_current':2.0,'avg_temp':38,'temp_rise':28,'cycle_duration':2800,'Re':0.145,'Rct':0.25,'delta_capacity':-0.15}),
]
print('='*70)
print(f'{"Scenario":<25}  {"SOH":>7}  {"RUL":>8}  State')
print('='*70)
for label, feat in tests:
    r = registry.predict(feat)
    print(f'{label:<25}  {r["soh_pct"]:6.2f}%  {r["rul_cycles"]:7.1f}  {r["degradation_state"]}')
print('='*70)
print()
print('Testing per-model RF prediction:')
for label, feat in tests:
    r = registry.predict(feat, 'random_forest')
    print(f'  RF {label:<22}  SOH={r["soh_pct"]:6.2f}%  RUL={r["rul_cycles"]:7.1f}')
