---
id: CONSISTENCY-COMMITMENT-LIFECYCLE
version: 1.1.0
language: ru
access_scope: architect
status: replaced_analytic_model_not_activated
updated_at: '2026-10-20'
---

# Жизненный цикл условий: указатель на уточнённую модель

Прежняя единая цепочка для предложения, условия, включения и ратификации заменена в документации [agreement_lifecycle.md](agreement_lifecycle.md). Этот файл сохранён для существующих ссылок и не содержит конкурирующей таблицы переходов.

В уточнённой модели proposed/withdrawn/superseded описывают предложение, conditionally_accepted/accepted — отдельное согласие участника с точным условием, included_in_protocol — связь включения, ratified — документ, active — обязательство. rejected всегда требует указать объект и область отказа. У этих объектов разные циклы; нельзя передавать одно поле status между ними.

Отзыв без замены — withdrawn. Для superseded требуется конкретная замена. accepted не доказывает активацию, а active не доказывает исполнение. Пересмотр ratified-соглашения не поддерживается до отдельного решения DEC-04; доработка отклонённого проекта не создаёт такого права.

Основания, исправленные исторические формулировки и нерешённые вопросы — [decision_register.yaml](decision_register.yaml), [consistency_resolution_proposal.md](consistency_resolution_proposal.md) и [consistency_audit.md](consistency_audit.md). Новая модель — аналитическое предложение, не утверждённый регламент приложения.
