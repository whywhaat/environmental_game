from lfx.custom import Component
from lfx.io import IntInput, Output
from lfx.schema.data import Data


class GameSettings(Component):
    display_name = "Настройки переговоров"
    description = "Три компонента проекта и число раундов; стоимость считается как сумма."
    icon = "sliders"
    inputs = [
        IntInput(name="water_cost", display_name="Вода и тарифы, млн ₽", value=50, required=True),
        IntInput(name="monitoring_cost", display_name="Мониторинг, млн ₽", value=20, required=True),
        IntInput(name="reconstruction_cost", display_name="Реконструкция, млн ₽", value=200, required=True),
        IntInput(name="max_rounds", display_name="Максимум раундов", value=15, required=True),
    ]
    outputs = [Output(name="settings", display_name="Game Settings", method="build_settings")]

    def build_settings(self) -> Data:
        return Data(data={
            "max_rounds": int(self.max_rounds),
            "components": [
                {"id": "water_supply", "name": "Вода и заморозка тарифов", "cost": int(self.water_cost) * 1_000_000},
                {"id": "independent_monitoring", "name": "Независимый мониторинг", "cost": int(self.monitoring_cost) * 1_000_000},
                {"id": "reconstruction", "name": "Реконструкция", "cost": int(self.reconstruction_cost) * 1_000_000},
            ],
        })
