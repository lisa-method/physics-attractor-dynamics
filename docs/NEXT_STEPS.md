# План продолжения текущего проекта

> На 2026-09-04 основной исследовательский цикл TODO 1–16 завершён;
> результаты собраны в [итоговом отчёте](FINAL_REPORT.md). Ниже сохранена история
> плана и возможные самостоятельные расширения. Новые эксперименты не обязательны
> для завершения текущего проекта и не запускаются автоматически.

## Цель

Довести восстановленный canvas-симулятор до одного из двух честных результатов:

1. **интерпретируемая модель:** найдена компактная формула/state machine,
   воспроизводящая обычное движение и редкие boundary-режимы;
2. **качественный прогноз:** если редкий режим нельзя полностью выразить простой
   формулой, его causal selector обучается с помощью ML, а координаты по-прежнему
   рассчитываются найденными детерминированными уравнениями.

Предпочтительная архитектура не противопоставляет эти варианты:

```text
наблюдаемое состояние + история режима
                |
                v
       rule-based или ML gate
                |
                v
 free / vertical / horizontal / period-2 / other
                |
                v
     детерминированная формула перехода
                |
                v
     координаты, углы и distance следующего шага
```

Black-box regression следующих координат остаётся контрольным baseline, а не
основной моделью: она хуже объясняет механизм и может скрыть редкие ошибки за
высокой общей accuracy.

## Текущая точка отсчёта

- Exact-formula hybrid one-step воспроизводит весь coordinate+angle+distance
  state numerical-exact в `98.69%`, а в material tolerance — в `99.84%`
  переходов.
- Standard collision selector имеет `100%` accuracy на уже объяснённых режимах.
- Медианный первый material failure в recursive rollout вырос `89 → 189`.
- Coordinate RMSE hybrid: `4.09` на горизонте 100 и `61.66` на горизонте 500
  против `8.60` и `95.35` у TODO 8.
- На полном сегменте hybrid `158.84` лучше и TODO 8 `169.31`, и
  constant-position baseline `169.25`.
- Главный необъяснённый остаток — causal выбор редкого special-режима: точная
  формула угла и period-2 continuation уже восстановлены.
- Условие reset неизвестно; границы эпизодов пока считаются внешними.

Все сегменты уже использовались в forensic discovery, а старый test был
скомпрометирован. Поэтому дальнейшие числа являются leave-one-segment-out
оценкой переноса, а не pristine test score.

## Статус выполнения на 2026-09-02

- Causal event dataset построен для 73 328 object-transitions; target-derived
  поля используются только для post-hoc labels и метрик.
- Period-2 переопределён как двухфазный state, а не только nonstandard branch:
  causal continuation получило `precision=1.000`, `recall=0.978`, `F1=0.989`.
- Shallow decision tree не прошёл special gate: `F1=0.636`.
- Fixed residual Random Forest прошёл event-level gate: `F1=0.881`.
- Special angle почти всегда относится к `k*90° ± 11.43°`; LOSO snap даёт
  special-angle MAE `0.12°`, но не numerical-exact значение.
- Полный sequential OOF one-step улучшил material complete-state share
  `97.63% → 99.81%` на всех 11 сегментах и maximum-angle MAE
  `0.130° → 0.012°`.
- Strict exact share ухудшилась `96.35% → 96.21%`, поэтому gate `98.17%` не
  пройден и recursive rollout по этому протоколу не запускался.

### TODO 10: выполнено

- Для 1 146 обычных special wall-событий найдено numerical-exact тождество
  `tan(theta) * cos(phi) = 2/10 = 0.2`, где `phi` — расстояние `free_angle` до
  ближайшей оси. Максимальная ошибка `theta` около `5.3e-8°`.
- LOSO Random Forest из TODO 9 оставлен неизменным; менялась только формула
  special-angle.
- Exact sequential one-step вырос `96.21% → 98.69%` и прошёл gate `98.17%`.
- Material one-step вырос `99.81% → 99.84%`; осталось 58 ошибок из 36 664.
- По условию протокола recursive rollout открыт. Медианный material horizon
  равен 189, RMSE ниже TODO 8 на горизонтах 100/500, а full-horizon RMSE ниже
  constant-position baseline. Все заранее зафиксированные rollout gates прошли.

### TODO 11: выполнено, acceptance gate не пройден

- Проверена двухпороговая stateful policy: высокий `T_enter` включает residual
  special-state, более низкий `T_stay` удерживает собственное предыдущее
  predicted решение.
- Для каждого outer сегмента пороги выбирались без его labels: 55 pairwise
  inner RF исключали одновременно outer и inner сегменты.
- Выбраны `T_enter=0.55–0.60` и `T_stay=0.30–0.45`; residual F1 вырос
  `0.8810 → 0.9159`.
- Full sequential one-step exact share выросла `98.6935% → 98.7508%`, material
  share — `99.8418% → 99.8664%`; exact улучшилась на 9 из 11 сегментов.
- Material failures уменьшились `58 → 49` (`-15.5%`), но pre-registered gate
  требовал не более 40. Остались 24 false positive, 24 false negative и один
  segment-start hidden-state переход.
- Поэтому TODO 11 не заменяет TODO 10, а recursive rollout не открыт. Это не
  отсутствие эффекта: эксперимент подтвердил hysteresis, но показал, что одной
  вероятности RF недостаточно для надёжного различения входа и продолжения.

### TODO 12/13: автомат проверен, TODO 11 остаётся лучшим selector

- Target-state разделён на 1 013 `entry`, 135 `continuation`, 1 014 `exit` и
  отдельный двухфазный `PERIOD2`. Два hidden segment-start special исключены из
  classifier training, но остаются в полном material score.
- TODO 12 обучал два независимых fixed-architecture RF-head. Nested selection
  впервые считался на полном teacher-forced sequential проходе
  `object1 → predicted distance → object2`, а run matching использовал IoU 0.5.
- Полностью раздельный автомат оказался хуже TODO 11: residual F1 `0.9030`,
  exact `98.6744%`, 65 material failures. Run F1 при этом `0.8997`, median
  onset/exit delay 0.
- TODO 13 сохранил общий TODO 11 RF для `NORMAL → SPECIAL` и применил отдельный
  head только для continuation/exit. Он вернулся к 49 material failures, но
  exact `98.7481%` и F1 `0.91549` на один transition/несколько десятитысячных
  хуже TODO 11 (`98.7508%`, `0.91586`).
- Continuation head сам по себе качественный: recall `0.9630`, correct-exit
  share `0.9980`. Но он заменяет только две ошибки противоположного знака и не
  даёт net gain. TODO 11 выигрывает как более простая модель.
- Gates TODO 12/13 не пройдены, recursive rollout не запускался. Оба результата
  сохраняются как ablations, а не удаляются или подгоняются.

### Следующий выбор

TODO 14 выполнил отдельный forensic-анализ без переобучения. Все 24 обычных FN —
пропущенные entry, 22 из 24 FP — ложные entry из NORMAL. `60.4%` RF-ошибок
находятся в пределах `|margin| ≤ 0.10`, но `39.6%` уверенные. TODO 15 затем
проверил простое решение «дать модели больше строк»: flat causal windows
`K=2/4/8` при замороженных geometry/PERIOD2/continuation. Оно не сработало:
лучшее `K=2` дало 125 failures и F1 `0.8777` против 49 и `0.9162` у TODO 11;
улучшились только 2/11 сегментов. Более длинные окна были ещё хуже.

### TODO 16: control и compact dynamics проверены

Один entry-head без нового блока (`current_only`) сравнен с тем же head плюс
31 compact dynamics feature. Одинаковы population, RF-конфигурация и nested
sequential LOSO. В обоих вариантах текущие обучающие признаки object2 строятся
через fixed base/PERIOD2 teacher object1, а не текущую truth-ветвь. Это делает
сравнение внутри TODO 16 парным, но не позволяет считать current_only точной
репликой старого TODO 15 без lag-блока.

- Control: 62 material failures, exact `98.6881%`, F1 `0.9053`.
- Compact dynamics: 86 material failures, exact `98.6935%`, F1 `0.9064`.
- TODO 11: 49 failures, exact `98.7508%`, F1 `0.9162`.
- Dynamics добавил только два exact-перехода против control, но 24 заметные
  ошибки. По material он лучше control на 1 сегменте, равен на 3, хуже на 7.
- Все promotion gates не пройдены. Clean run 76/76 cells, 0 errors; 5 быстрых
  feature tests проходят. Нового recursive rollout нет.

### Следующий выбор после TODO 16

1. Сохранить TODO 11 как лучший stateful selector и TODO 10 как модель с
   проверенным rollout; не расширять окно и не менять пороги задним числом.
2. Зафиксировать отрицательные ablations, но не объявлять отсутствие закона:
   провал конкретного RF не доказывает ненаблюдаемость или случайность режима.
3. Для пути «формула» вернуться к точечному rule-discovery на frozen 49 ошибках
   TODO 11: проверять геометрическое условие переключения и перенос правила
   между сегментами; новые гипотезы регистрировать до оценки.
4. Для пути «прогноз» следующим отдельным протоколом рассмотреть оценку
   неопределённости selector и отказ от уверенного выбора на неоднозначных
   шагах. Ensemble/sequence-модель не запускаются автоматически.
5. Новую модель допускать к recursive проверке только по прежним gates.
   Для окончательного подтверждения нужен новый независимый набор эпизодов;
   текущие 11 сегментов уже исследованы. Reset остаётся внешней границей.

## Этап 1. Собрать causal event dataset

Для каждого перехода и каждого объекта сформировать одну строку с признаками,
которые доступны **до** предсказываемого шага:

- текущие координаты и угол;
- free proposal и величина overshoot по каждой оси;
- сторона стены и индикатор corner;
- длина текущего boundary-run;
- предыдущие одна–три выбранные ветви;
- время с последнего свободного шага и столкновения;
- causal lag-1/lag-2 differences;
- period-2 indicator, рассчитанный только по прошлым состояниям;
- для object 2 — вычисленные simulator координаты object 1 и промежуточный
  `distance`, но не truth из следующей строки.

Target размечается post-hoc по ближайшему фактически наблюдавшемуся переходу:

```text
free
vertical reflection
horizontal reflection
period-2 wall loop
other special
```

Reset не смешивается с этими классами.

Проверки этапа:

- отсутствие future features;
- mutually exclusive labels;
- counts по классам, объектам, сегментам и длинам непрерывных event-runs;
- визуальная проверка нескольких переходов каждого класса.

## Этап 2. Сначала искать формулу/state machine

Проверить по очереди простые гипотезы:

1. ветвь определяется стороной стены, знаком угла и overshoot;
2. ветвь чередуется по чётности `boundary-run`;
3. состояние period-2 задаётся предыдущей ветвью или фазой `0/1`;
4. выход из loop определяется новым free proposal, углом или расстоянием до
   corner;
5. разные объекты используют одинаковый автомат с разными входами;
6. сохранённое скрытое состояние переносится между последовательными
   столкновениями.

Инструменты поиска: contingency tables, небольшие decision trees, truth tables и
последовательная проверка предлагаемых тождеств до numerical tolerance. Decision
tree здесь используется прежде всего как способ подсказать короткое правило.

Формула принимается только если она:

- causal;
- работает не на одном loop, а leave-one-segment-out;
- использует одинаковые правила для обоих объектов либо явно объясняет разницу;
- улучшает не только общую accuracy, но и recall редких special transitions;
- улучшает recursive rollout.

## Этап 3. Если короткой формулы недостаточно — residual ML gate

ML решает только задачу выбора режима. Последующий переход рассчитывается уже
известными формулами.

Порядок моделей:

1. majority/known-rule baseline;
2. shallow decision tree;
3. regularized multinomial logistic regression;
4. random forest или histogram gradient boosting;
5. sequence model — только если табличные модели докажут, что короткой истории
   недостаточно.

Классы несбалансированы, поэтому primary metrics — per-class recall, macro-F1 и
confusion matrix. Общая accuracy вторична: текущая детерминированная модель уже
объясняет более 96% переходов без ML.

Probability модели можно использовать для режима `unknown`: при низкой
уверенности simulator сообщает, что надёжный следующий переход не определён,
вместо молчаливого физически неверного прогноза.

## Этап 4. Честное разделение и подбор

- Единица split — целый reset-сегмент.
- Все строки одного непрерывного wall-run остаются в одном fold.
- Основная схема — leave-one-segment-out; соседние строки случайно не
  перемешиваются.
- Подбор depth, regularization, class weights и probability threshold делается
  только во внутренних folds.
- Итоговый отчёт показывает median, диапазон и каждый сегмент отдельно.

Такой протокол проверяет перенос правила на другой эпизод, хотя полностью blind
test для этого уже восстановить нельзя.

## Этап 5. Проверка one-step модели

Для каждого held-out сегмента считать:

- branch macro-F1 и recall каждого special режима;
- exact coordinate rate;
- exact full-state rate;
- MAE/RMSE координат, углов и `distance`;
- метрики отдельно для free, standard collision, period-2 и other-special;
- coverage и ошибку для confident/unknown predictions.

Предлагаемый основной gate относительно текущей ошибки:

- сократить full-state one-step error хотя бы вдвое: с `3.65%` до не более
  `1.83%`, то есть получить не менее `98.17%` exact state;
- улучшить результат минимум на 8 из 11 held-out сегментов;
- получить special-mode macro-F1 не ниже `0.80`.

Числа нужно считать предварительно зафиксированными ориентирами; если
редкий класс отсутствует в конкретном fold, его метрика там отмечается как `NA`,
а не как ноль или единица.

## Этап 6. Recursive rollout

Новая модель запускается без доступа к будущим truth-строкам. Для каждого
сегмента сравниваются:

1. constant-position baseline;
2. текущий TODO8 simulator;
3. новый formula/state-machine simulator;
4. hybrid simulator с ML gate;
5. optional direct-coordinate ML baseline.

Метрики:

- распределение первого exact failure;
- распределение первого material failure;
- coordinate RMSE на горизонтах `10`, `50`, `100`, `500` и до конца сегмента;
- доля rollout, остающихся finite и внутри допустимой области;
- тип события, с которого началось расхождение.

Предлагаемый rollout gate:

- медианный material-failure horizon не меньше `178` update — вдвое больше
  текущих `89`;
- RMSE ниже текущего TODO8 simulator на горизонтах 100 и 500;
- на полном сегменте модель должна быть лучше constant-position baseline, иначе
  заявляется только short/medium-horizon usefulness.

## Этап 7. Отдельно решить вопрос reset

Сначала проверить, существуют ли перед reset наблюдаемые causal precursors. Если
они есть, построить отдельную hazard/classification модель `reset within k
steps`. Если устойчивых предвестников нет, reset признаётся внешним событием:

- simulator получает episode boundary извне;
- качество движения оценивается только внутри эпизода;
- невозможность предсказать внешний reset не считается ошибкой закона движения.

Нельзя обучать координатную модель перескакивать через reset: это смешивает два
разных процесса и создаёт фиктивную динамику.

## Этап 8. Финальный научный результат

В зависимости от эксперимента проект завершается одним из трёх честных выводов:

1. **Формула найдена:** опубликован компактный causal state machine и его
   исполняемый simulator.
2. **Hybrid result:** основная кинематика и геометрия заданы формулами, а ML
   надёжно выбирает редкие ветви и улучшает rollout.
3. **Predictability limit:** доступных полей и короткой истории недостаточно для
   редкого режима; показан измеренный горизонт предсказуемости и локализована
   скрытая переменная.

Во всех трёх случаях нужно явно писать, что восстановлен алгоритм синтетической
гибридной системы, а не универсальный закон механики реальных тел.

## Ближайший исполняемый блок

1. Построить и проверить causal event dataset.
2. Провести аудит period-2 и boundary-run признаков.
3. Обучить shallow decision tree только как rule-discovery probe.
4. Выписать найденное дерево как несколько проверяемых логических гипотез.
5. Сравнить formula gate и tree gate в leave-one-segment-out one-step режиме.
6. Только после прохождения one-step gate запускать recursive rollout.
