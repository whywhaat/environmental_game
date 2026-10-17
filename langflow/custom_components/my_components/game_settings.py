from lfx.custom import Component
from lfx.io import BoolInput, IntInput, Output
from lfx.schema.data import Data


class GameSettings(Component):
    display_name = "Настройки сценария игры"
    description = "Параметры экологического сценария: целевой бюджет, сроки и переговорные условия."
    icon = "sliders"

    inputs = [
        IntInput(
            name="total_cost",
            display_name="Целевой бюджет (млн руб.)",
            value=1000,
            info="Общая требуемая стоимость реконструкции очистных сооружений.",
            required=True,
        ),
        IntInput(
            name="default_timeline",
            display_name="Базовый срок завода (мес.)",
            value=36,
            info="Стартовый срок модернизации, предлагаемый промышленным холдингом.",
            required=True,
        ),
        IntInput(
            name="water_supply_cost",
            display_name="Стоимость подвоза воды (млн руб.)",
            value=50,
            info="Расходы администрации города на обеспечение питьевой водой (вычитаются из её лимита).",
            required=True,
        ),
        IntInput(
            name="monitoring_cost",
            display_name="Стоимость эко-мониторинга (млн руб.)",
            value=20,
            info="Расходы региональной власти на независимые лаборатории (вычитаются из её лимита).",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="settings",
            display_name="Game Settings",
            method="build_settings",
            types=["Data"],
        )
    ]

    def build_settings(self) -> Data:
        total_cost_rub = float(getattr(self, "total_cost", 1000) or 1000) * 1_000_000.0
        timeline = int(getattr(self, "default_timeline", 36) or 36)
        water_cost_rub = float(getattr(self, "water_supply_cost", 50) or 50) * 1_000_000.0
        mon_cost_rub = float(getattr(self, "monitoring_cost", 20) or 20) * 1_000_000.0

        levers = [
            {
                "id": "water_supply",
                "title": "Гарантия питьевой воды жителям",
                "aliases": ["вода", "водоснабжение", "питьевая вода", "water"],
                "type": "bool",
                "owner": "Администрация города N",
                "cost": water_cost_rub,
                "default": False,
                "description": "Внебюджетная поставка чистой питьевой воды жителям моногорода до ввода сооружений и заморозка тарифов ЖКХ",
            },
            {
                "id": "timeline",
                "title": "Срок реконструкции (мес.)",
                "aliases": ["срок", "время", "timeline", "months"],
                "type": "int",
                "owner": "Промышленный холдинг",
                "cost": 0.0,
                "default": timeline,
                "min_value": 24,
                "max_value": 48,
                "description": "Срок ввода пусковых комплексов очистных сооружений в эксплуатацию",
            },
            {
                "id": "independent_monitoring",
                "title": "Независимый экологический мониторинг",
                "aliases": ["мониторинг", "контроль", "экоконтроль", "monitoring"],
                "type": "bool",
                "owner": "Региональная власть",
                "cost": mon_cost_rub,
                "default": False,
                "description": "Привлечение независимых аккредитованных лабораторий и ежемесячная открытая публикация результатов замеров",
            },
            {
                "id": "public_council",
                "title": "Общественный контроль и комиссия",
                "aliases": ["совет", "комиссия", "общественный совет", "council"],
                "type": "bool",
                "owner": "Экологическая организация (НКО)",
                "cost": 0.0,
                "default": False,
                "description": "Включение представителей инициативной группы жителей и эко-НКО в приемочную комиссию и совет проекта",
            },
        ]

        return Data(
            data={
                "total_cost": total_cost_rub,
                "levers": levers,
            }
        )

