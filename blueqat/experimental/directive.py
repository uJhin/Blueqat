from blueqat import BlueqatGlobalSetting
from blueqat.gate import BackendSpecificGate

BlueqatGlobalSetting.register_gate('sp', BackendSpecificGate)
