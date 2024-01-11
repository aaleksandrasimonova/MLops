### Системная конфигурация
Microsoft Windows 10

Версия	22H2

Сборка ОС	19045.3803

vCPU: 5, RAM: 8GB

### Описание решаемой задачи

Рассмотрена задача классификации рукописных цифр (mnist). Исходными данными являются изображения рукописных цифр 28x28.

### Дерево model репозитория

    triton/model_repository
    └── mnist_model
        └── 1
            └── model.onnx
        └── config.pbtxt
 

### Метрики до оптимизаций

    Throughput: 1413.22 infer/sec
    Avg latency: 2118 usec (standard deviation 644 usec)
    p50 latency: 2183 usec
    p90 latency: 2602 usec
    p95 latency: 2777 usec
    p99 latency: 3896 usec
    Avg HTTP time: 2098 usec (send/recv 317 usec + response wait 1781 usec)


### Метрики после оптимизаций

    Throughput: 2462.18 infer/sec
    Avg latency: 1217 usec (standard deviation 3028 usec)
    p50 latency: 1248 usec
    p90 latency: 1358 usec
    p95 latency: 1395 usec
    p99 latency: 1695 usec
    Avg HTTP time: 1213 usec (send/recv 71 usec + response wait 1142 usec)


### Мотивация выбора

С увеличением количества instance с 1 до 3 наблюдалась улудшение параметров throught и latency, при дальнейшем увеличении параметры пошли на спад. Далее производился подбор параметров max_queue_delay_microseconds  и была выяснено, что лучшие показатели достигаются при значение 400.

Далее приведена таблица результатов экспериментов.


| count |  max_queue_delay_microseconds | throughput |p50 latency | p90 latency | p95 latency | p99 latency |
|:-----:|:-----------------------------:|:----------:|:----------:|:-----------:|:-----------:|:-----------:|
|   1   |                          300  |   1413.22  |      2183  |       2602  |       2777  |       3896  |
|   2   |                          300  |   1844.59  |      1481  |       2381  |       2596  |       3204  |
|   3   |                          300  |   2015.94  |      1207  |       2435  |       2656  |       3491  |
|   4   |                          300  |   1352.8   |      2251  |       2667  |       2842  |       4039  |
|   5   |                          300  |   1358.45  |      1547  |       3973  |       5567  |       9809  |
|   3   |                          100  |   1536.37  |      1999  |       2357  |       2522  |       3679  |
|   3   |                          200  |   2382.42  |      1202  |       1635  |       1873  |       2290  |
|   3   |                          400  |   2462.18  |      1248  |       1358  |       1395  |       1695  |
|   3   |                          500  |   2235.11  |      1334  |       1516  |       1846  |       2120  |




