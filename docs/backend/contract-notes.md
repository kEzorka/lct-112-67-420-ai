# Заметки о контракте с Java-бэкендом

Дата: 28.09.2026. Источник: приватный репозиторий `fluffb4ll/lct-112-67-420-http-backend`, ветка по умолчанию, коммит `5757c89`. Бэкенд прочитан целиком, только для чтения.

Код бэкенда сюда **не копируется**: ниже — пути к файлам, названия сущностей, полей и эндпоинтов, достаточные для согласования контрактов. Пути даны относительно корня бэкенда; `…/` = `src/main/java/com/fluffb4ll/lct112HttpBackend/`.

## 0. Коротко

- Бэкенд сейчас — **сервис учётных записей и аудита**: логин по cookie, CRUD пользователей для администратора, роли с правами, отделы, учебные группы, журнал действий.
- Учебного домена **ещё нет**: нет карточки, попытки, событий попытки, снимка версий, сценариев, эталонов, рубрики, критериев, оценок, маршрутизации, медиа.
- **Вызовов ИИ нет.** Подключены стартеры gRPC-клиента и WebSocket, но не используются.
- Схема БД в репозитории не хранится (нет миграций), Hibernate только валидирует её при старте.

Следствие: раздел 4.2 [`docs/ai/agent-prompt.md`](../ai/agent-prompt.md) описывает контракты, которых на стороне бэкенда пока нет. По правилу промпта ИИ-контур описывает их черновиком в `contracts/` и мокает.

## 1. Стек, сборка, запуск

| Пункт | Значение | Где |
|---|---|---|
| Java | 25 (`java.version`) | `pom.xml` |
| Фреймворк | Spring Boot 4.1.1: webmvc, data-jpa, validation, websocket, grpc-client; `spring-security-crypto` (без Spring Security фильтров); Lombok | `pom.xml` |
| JSON | Jackson 3 (`tools.jackson.*`), `default-property-inclusion: non_null` — поля со значением `null` в ответах опускаются | `src/main/resources/application.yaml`, `…/service/AuditService.java` |
| БД | PostgreSQL (драйвер `org.postgresql`), Hikari: пул 15, min idle 5, таймаут подключения 5 с; `ddl-auto: validate`, `open-in-view: false`, batch 100 | `application.yaml` |
| Сборка | Maven Wrapper (Maven 3.9.16): `./mvnw clean package` | `mvnw`, `.mvn/wrapper/maven-wrapper.properties` |
| Образ | `eclipse-temurin:25-jre-alpine`, непривилегированный пользователь, порт 8080, G1GC, `MaxRAMPercentage=75` | `Dockerfile` |
| Скрипт | Windows `build.bat`: сборка без тестов + `docker buildx` под `linux/arm64` и push в GHCR | `build.bat` |
| Тесты | Один `contextLoads` (требует доступной БД) | `src/test/java/.../Lct11267420HttpBackendApplicationTests.java` |
| Точка входа | `@SpringBootApplication` + `@EnableScheduling` | `…/Lct11267420HttpBackendApplication.java` |

**Переменные окружения:** `DB_URL`, `DB_USER`, `DB_PASSWORD`, `PEPPER` (обязателен — без него энкодер паролей не создаётся), `SADMIN_USERNAME` (по умолчанию `admin`), `SADMIN_PASSWORD` (если пусто или не проходит проверку — генерируется и выводится в лог один раз), `TOKEN_CLEANUP_DELAY` (30m), `TOKEN_CLEANUP_INIT_DELAY` (1m). Время жизни сессии — 1 ч (`app.security.auth.token-expiration-*`).

**Условия старта:** в БД должна существовать схема с таблицами из раздела 3 и роль с именем `ROLE_ADMIN`, иначе приложение падает при инициализации суперадмина (`…/config/SuperAdminInitializer.java`).

**Для локального стенда:** cookie сессии выставляется с флагом `Secure`, т. е. вне `localhost` нужен HTTPS (согласуется с D-055).

## 2. Сущности и поля

### 2.1. Что есть

| Сущность | Таблица | Поля | Файл |
|---|---|---|---|
| Пользователь | `iam.users` | `id` UUID; `username` ≤64, уникален; `password_hash` ≤60 (хэш пароля); `full_name` ≤150; `role_id` → roles (обязателен); `department_id` → departments (необязателен); `created_at`, `updated_at`; `is_active`; `must_change_password`; M:N с учебными группами через `iam.study_group_members(student_id, group_id)` | `…/entity/UserEntity.java` |
| Роль | `iam.roles` | `id` int (identity); `name` ≤50, уникально; `permissions` `integer[]` — индексы из перечисления прав | `…/entity/RoleEntity.java` |
| Права | — (enum) | `ADMIN_CAN_EDIT_USERS`=0, `ADMIN_CAN_EDIT_ADMINS`=1, `ADMIN_CAN_READ_USERINFO`=2. Других прав нет | `…/model/enums/Permissions.java` |
| Отдел | `iam.departments` | `id` UUID; `name` ≤150, уникально; `code` ≤50, уникален | `…/entity/DepartmentEntity.java` |
| Учебная группа | `iam.study_groups` | `id` UUID; `name` ≤150, уникально; `teacher_id` UUID (без связи на уровне JPA); `created_at` | `…/entity/StudyGroupEntity.java` |
| Сессионный токен | `iam.auth_tokens` | `id` UUID; `user_id`; `token` UUID, уникален; `token_type` ≤20 (`SESSION`); `expires_at`; `created_at` | `…/entity/AuthTokenEntity.java` |
| Запись аудита | `audit.action_logs` | `id` bigint identity; `user_id` (кто сделал, может быть пусто); `event_type` ≤100; `entity_name` ≤50; `entity_id` UUID; `old_value`, `new_value` jsonb; `ip_address` ≤45 (с учётом `X-Forwarded-For`); `created_at` | `…/entity/ActionLogEntity.java`, `…/service/AuditService.java` |

Перечисления аудита заранее содержат учебные объекты, которых ещё нет:
- `EntityType`: `USER, ROLE, STUDY_GROUP, DEPARTMENT, SCENARIO, OPERATOR_CARD, EVALUATION, AUTH_TOKEN` — `…/model/enums/EntityType.java`;
- `EventType`: `LOGIN, USER_CREATION, USER_DELETION, USER_UPDATE, SCENARIO_CREATION, SCENARIO_DELETION, SCENARIO_UPDATE, SESSION_EVALUATION` — `…/model/enums/EventType.java`. Фактически пишутся только `USER_CREATION`, `USER_DELETION`, `USER_UPDATE`.

### 2.2. Пользователи и роли — как работает сейчас

- Роли и их права лежат в БД; в коде по имени известна только `ROLE_ADMIN`. Роль с `id = 0` считается «админской»: её назначение/изменение требует `ADMIN_CAN_EDIT_ADMINS`, остальных — `ADMIN_CAN_EDIT_USERS` (`…/service/UserUpdateService.java`).
- Роли преподавателя и диспетчера в коде не упоминаются; прав для них нет.
- Преподаватель связан с группой только полем `study_groups.teacher_id`; ученики — через `study_group_members`.
- Удаление пользователя — физическое (в коде есть TODO про мягкое удаление).
- Самоудаление, самодеактивация и выставление себе `must_change_password` запрещены.

### 2.3. Чего нет (из списка задания)

| Объект | Статус в бэкенде |
|---|---|
| `Card` (`source_payload`, `dispatcher_edits`, `card_revision[]`) | Нет. Есть только значение `OPERATOR_CARD` в перечислении аудита |
| `TrainingAttempt` (`events`, `evidence`, `status`) | Нет |
| События попытки (`notification_shown`, `card_opened`, …) | Нет. Журнал `audit.action_logs` — административный аудит, не журнал попытки |
| `AttemptVersionSnapshot` | Нет |
| `ScoreVersion` | Нет. Есть только значения `EVALUATION` / `SESSION_EVALUATION` в перечислениях аудита |
| Рубрика, критерии, `rubric_version`, `criterion_status` | Нет |
| Сценарии, эталоны, версии | Нет. Есть только `SCENARIO_*` в перечислении аудита |

## 3. Схема БД и миграции

- Миграций (Flyway/Liquibase, SQL-скриптов) в репозитории **нет**. `ddl-auto: validate` — схему создаёт кто-то снаружи, приложение лишь проверяет соответствие сущностям.
- Ожидаемые схемы и таблицы (восстановлено из отображений JPA):
  - `iam`: `users`, `roles`, `departments`, `study_groups`, `study_group_members`, `auth_tokens`;
  - `audit`: `action_logs`.
- Типы, важные для совместимости: идентификаторы — UUID (кроме `roles.id` int и `action_logs.id` bigint), время — `timestamptz` (`OffsetDateTime`), JSON — `jsonb`, права — `integer[]`.
- Фоновая задача: удаление просроченных токенов раз в 30 мин (`…/engine/TokenCleanupScheduler.java`). Других задач очистки/хранения нет.

Нужно получить от владельца бэкенда DDL (или договориться о миграциях в репозитории), иначе ИИ-контур не может поднять совместимую БД для интеграционных тестов.

## 4. REST, очереди, WebSocket

Все ответы об ошибках — `{ errorMessage }`. Аутентификация — cookie `AUTH_TOKEN` (UUID, `HttpOnly`, `Secure`, `SameSite=Lax`, срок = срок токена); заголовков `Authorization` нет. Права проверяются в сервисах, не фильтрами. Коды: 401 — нет/неверный/просроченный токен или нет права (при просрочке cookie стирается), 409 — ошибка изменения пользователя (в том числе «не найден»).

Файлы: `…/controller/AuthController.java`, `…/controller/AdminController.java`, `…/controller/GlobalExceptionHandler.java`, DTO — `…/dto/request/`, `…/dto/response/`.

| Метод | Путь | Вход | Выход | Право |
|---|---|---|---|---|
| POST | `/api/auth/login` | `{username, password}` | 200 + `Set-Cookie: AUTH_TOKEN`; тело `UserInfo` | — |
| GET | `/api/auth/logout` | cookie | 200, cookie стирается | — |
| POST | `/api/admin/users/create` | `{username, password, fullName, roleId, departmentId?, mustChangePassword}` | 201, `Location: /api/admin/users/{id}`, `{userId}` | `ADMIN_CAN_EDIT_USERS` / `…_ADMINS` по роли |
| POST | `/api/admin/users/update` | `{userId, username?, password?, fullName?, roleId?, departmentId?, removeDepartment?, isActive?, mustChangePassword?}` | 200 | то же |
| DELETE | `/api/admin/users/{uuid}` | — | 200 | то же |
| GET | `/api/admin/users/{uuid}` | — | `UserInfo` | `ADMIN_CAN_READ_USERINFO` |
| GET | `/api/admin/users?page=0&size=20` | `size` ограничен 1–100, сортировка по `fullName` | `{content[], page, size, totalElements, totalPages}`, строка: `{id, fullName, roleName, departmentName, isActive}` | `ADMIN_CAN_READ_USERINFO` |

`UserInfo` = `{id, username, fullName, role, permissions[], department{id, code, name}, studyGroups[{id, name, teacherId}]}`.

Особенность `users/update`: `departmentId` применяется, только если `removeDepartment` явно `false`, иначе молча игнорируется.

Проверки формата: логин — латиница/цифры/`_.-`, 1–16 символов; пароль — ≥8, минимум одна буква и одна цифра; ФИО — кириллица/латиница, пробел, апостроф, ≤150 (`…/util/RegexSecurityUtil.java`). ФИО с дефисом (двойные фамилии) и с буквой «ё» не проходит — учесть при генерации тестовых пользователей.

**Очереди:** нет. **WebSocket:** зависимость есть, конфигурации и обработчиков нет. **gRPC:** клиентский стартер есть, `.proto`, стабов и настроек каналов нет. Асинхронность: аудит помечен как асинхронный, но асинхронное выполнение в приложении не включено, так что запись идёт в том же потоке, в отдельной транзакции.

## 5. Вызовы ИИ

Нет ни одного: STT/TTS/LLM, генерация сценариев и карточек, оценивание, рекомендации в бэкенде не вызываются и не описаны. Единственный намёк на будущую интеграцию — стартер gRPC-клиента в `pom.xml`. Промпт ИИ-контура по умолчанию предполагает локальный HTTP/WebSocket-сервис (M0) — протокол нужно согласовать (см. «Расхождения», п. 13).

## 6. Расхождения

Сравнение с разделом 4 `docs/ai/agent-prompt.md` и W-01/W-02 `docs/architecture/decision-register.md`. Промпт не правился.

| # | Пункт | В промпте | В бэкенде | Предложение |
|---|---|---|---|---|
| 1 | Объём бэкенда | 4.2: бэкенд даёт доменную модель, журнал событий, снимок версий, `ScoreVersion`, маршрутизацию, медиатранспорт, жизненный цикл, расчёт балла, RBAC | Только IAM (пользователи, роли, отделы, группы, токены) и административный аудит | Подтвердить с backend-участником, кто владеет каждым пунктом 4.2 и в каком порядке они появятся; до этого — черновики в `contracts/` + моки (как требует промпт) |
| 2 | `Card` | `{source_payload, dispatcher_edits, card_revision[]}` в PostgreSQL | Нет; в аудите тип `OPERATOR_CARD` | Завести таблицу карточки с неизменяемым `source_payload` и append-only ревизиями; переименовать `OPERATOR_CARD` в нейтральное `INCIDENT_CARD` — обучаемый диспетчер ДДС, не оператор 112 (D-001) |
| 3 | Терминология | Обучаемый — диспетчер ДДС (D-001) | README описывает продукт как тренажёр для операторов 112 | Поправить README/перечисления, чтобы не закреплять неверную роль в API |
| 4 | `TrainingAttempt` и события | `notification_shown, card_opened, card_revision, selected_action, played_audio_ack, submit_clicked, submit_accepted, timeout`; тайм-аут — событие | Нет; `audit.action_logs` без `attempt_id`, порядкового номера, клиентских/серверных отметок, `ack_id`, `call_id` | Отдельный append-only журнал попытки (не `action_logs`): `attempt_id`, `seq`, `type`, `server_ts`, `client_ts`, `payload`, источник; эндпоинт/канал приёма событий от ИИ-worker (`ack.*`, сегменты STT, сбои моделей) |
| 5 | `AttemptVersionSnapshot` | 1:1 со стартом попытки: scenario · reference · card_schema · classifier · routing_rules · rubric · time_policy · model/template · difficulty_config | Нет | Таблица снимка, заполняется при старте; ИИ-worker отдаёт свои версии моделей/промптов в preflight |
| 6 | `ScoreVersion` | append-only, `parent_score_version_id`, `author`, `reason`, `criterion_results[]`, `total_score`, `provisional` | Нет; только `EVALUATION` / `SESSION_EVALUATION` в перечислениях аудита | Таблица версий оценки + таблица результатов критериев (`criterion_status`, значение, доказательство, версия модели/промпта) |
| 7 | Рубрика, W-01 | Группы 20/25/25/20/10, 0,5 только в двух случаях, порог 75, три критические ошибки; настраиваемо через `rubric_version` | Нет ни рубрики, ни критериев, ни `rubric_version` | Хранить рубрику как версионируемую конфигурацию (веса, правила 0,5, порог, список критических ошибок), а не в коде — W-01 заменяется ответом Q05 без изменения кода |
| 8 | RBAC (D-054, Q01) | Три роли; преподаватель утверждает сценарии и корректирует оценки; администратор оценки не меняет | Роли в БД, права только административные (3 шт.); по имени известна лишь `ROLE_ADMIN`; нет прав преподавателя/диспетчера | Добавить права: утверждение сценария/эталона, корректировка оценки (`ScoreVersion` N+1), пометка ошибки STT, назначение пула, просмотр аналитики группы; администратору не давать права на оценки |
| 9 | Профиль ДДС | Профиль/рекомендации/рейтинг — «по профилю ДДС» | `iam.departments` (`name`, `code`) у пользователя; смысл не описан | Уточнить, является ли `department` профилем ДДС. Если да — ссылаться на него из сценария и попытки; если нет — отдельный справочник |
| 10 | Удаление пользователя vs C-09 | Оценки, транскрипты, основания не перезаписываются и не теряются в пределах срока хранения; W-02: оценки бессрочно | Физическое удаление пользователя (TODO про мягкое) | Мягкое удаление/деактивация; попытки и `ScoreVersion` должны переживать удаление учётной записи (автор и ученик — по UUID) |
| 11 | W-02, сроки хранения | Карточки/попытки 1 год, аудио 30 дней, транскрипты 1 год, оценки бессрочно, backup ежедневно ×14 дней, журналы безопасности ≥6 мес. | Только очистка просроченных токенов; политики хранения, object store для аудио и задачи удаления нет | Категорийная политика хранения как конфигурация + задача удаления с аудируемой записью о недоступности (C-09); `audit.action_logs` не чистить раньше 6 месяцев |
| 12 | Аудит и ПДн (инвариант 9, D-055/D-058) | В логах — не больше необходимого | Состав полей `old_value`/`new_value` не зафиксирован | Allowlist полей аудита без секретов и лишних ПДн; согласовать с владельцем бэкенда |
| 13 | Транспорт ИИ | M0: локальный HTTP/WebSocket-сервис на Python | Стартер gRPC-клиента (не используется), WebSocket не настроен | Выбрать один протокол бэкенд ↔ inference-worker (gRPC или HTTP+WS) и описать в `contracts/`; для живого звонка и событий `ack.*` нужен потоковый канал |
| 14 | Межсервисная аутентификация | ИИ пишет только в разрешённые сущности (4.2) | Только пользовательская cookie-сессия, сервисной аутентификации нет | Сервисный токен или mTLS внутри локального контура; права ИИ-worker ограничены записью реплик, событий, замечаний, черновиков |
| 15 | Маршрутизация (D-016, C-03) | Детерминированный движок правил, `routing_rules_version`, `rule_status` | Нет | Уточнить владельца движка; ИИ использует только его выход |
| 16 | Медиатранспорт VoIP (D-021, D-023) | ИИ получает аудиопоток и события вызова (`call_id`, mute, hold, завершение) | Нет | Уточнить владельца и протокол медиатракта |
| 17 | Время (C-01) | Серверные и клиентские отметки, `time_policy_version` | Серверное время ставится приложением (`OffsetDateTime.now()`) только для пользователей/токенов/аудита | При проектировании журнала попытки хранить обе отметки и версию политики времени |
| 18 | Схема БД | PostgreSQL + object store + vector index (Miro 05) | PostgreSQL без миграций в репозитории; object store и vector index нет | Миграции в репозитории бэкенда; object store и vector index — решить, чьи (vector index для RAG логично держать в ИИ-контуре) |

Совпадает: PostgreSQL, UUID-идентификаторы, `timestamptz`, `jsonb` — можно использовать в контрактах ИИ без конвертаций.

## 7. Чего нет в бэкенде, но нужно ИИ

1. Хранение карточки: исходная карточка от ИИ-оператора 112, параметры вариации и seed, ручные правки диспетчера с ревизиями.
2. Попытка и её жизненный цикл, режим (`самостоятельный / делай как я / с поддержкой`), флаг технического нарушения, признак неполной сдачи (C-02).
3. Append-only журнал событий попытки и канал приёма событий от ИИ-worker (`ack.generated/sent_to_media/played_by_client/delivery_unconfirmed`, частичные и финальные результаты STT, реплики, подсказки, сбои моделей).
4. `AttemptVersionSnapshot` с версиями моделей, промптов и шаблонов ИИ.
5. Сценарии и эталоны: `scenario_version` + `reference_version`, жизненный цикл «черновик → проверка → утверждён → опубликован → архив», комментарии преподавателя, признак диагностического сценария и gate C-03.
6. Рубрика как версионируемая конфигурация (W-01) и хранилище результатов критериев с `criterion_status`, доказательством и версиями.
7. `ScoreVersion` (append-only) и экспертная корректировка с автором и причиной; пометка устаревших агрегатов (C-08).
8. Движок маршрутизации с `routing_rules_version`, `rule_status` и версией классификатора.
9. Медиатранспорт VoIP-эмулятора и события вызова.
10. Object store для аудио, транскриптов и документов базы знаний; политика хранения W-02 и задача удаления с записью о недоступности (C-09).
11. Транскрипты с интервалами и отдельные записи «ошибка распознавания» от преподавателя.
12. Профиль подготовки и рекомендации с версиями, снимком назначенного пула и действием преподавателя (C-08); назначение пула заданий преподавателем.
13. Данные для аналитики группы: связь попыток с учебной группой и профилем ДДС, фильтр по `valid_score`.
14. Права преподавателя и диспетчера (утверждение, корректировка, назначение, просмотр аналитики), сервисная аутентификация ИИ-worker.
15. Push-канал к клиенту (уведомление о карточке, реплики, статус оценки) — WebSocket не настроен.
16. Preflight перед стартом попытки и видимый администратору статус компонентов ИИ (журнал сбоев, метрики).
17. Протокол бэкенд ↔ inference-worker (gRPC или HTTP/WS) и его версия.
18. Миграции схемы БД в репозитории.
