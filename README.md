# GNARL — реализация архитектуры из статьи

Полная реализация архитектуры и MDP-формулировок из статьи
**"Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined
through Reinforcement Learning"** (Schutz, Darvariu, Panagiotaki, Lacerda,
Hawes).

Написано на **JAX** (не PyTorch — в песочнице, где это собиралось, диска не
хватило на CUDA-зависимости torch; CPU-сборка JAX самодостаточна и к тому же
соответствует собственному JAX-стеку авторов для NAR-бейзлайнов).

## Структура

```
gnarl/
  model.py            — архитектура Encode → Process → Act (Sec. 4.2, Appendix B):
                         энкодер признаков, MPNN-процессор без рекуррентности
                         между шагами MDP, proto-action актор (Darvariu et al.,
                         2021b) с маскированием действий, критик для PPO.
  utils.py             — генерация графов (ER, BA, Евклидов TSP).
  envs/
    base.py            — базовый класс GNARLEnv / GraphState (M = <S,A,T,R,h>).
    bfs_dfs.py          — BFS/DFS: Algorithm 1 (transition), Algorithms 9-10
                          (эксперты), Algorithm 8 (проверка корректности DFS).
    bellman_ford.py     — Bellman-Ford: Table 7, Algorithm 3, Algorithm 11.
    mst_prim.py         — MST-Prim: Table 10, Algorithm 4, Algorithm 12.
    tsp.py              — TSP: Definition 1, Table 11, Algorithm 5 (+ Held-Karp
                          как замена решателя Concorde).
    mvc.py              — MVC: Definition 2, Table 12, Algorithm 6 (+ точный
                          ILP через PuLP/CBC как замена решателя из статьи).
  training/
    bc.py               — Behavioural Cloning (Eq. 3): имитация экспертного
                          распределения действий.
    ppo.py              — PPO (Eq. 2) с маскированием действий и GAE — обучение
                          только по сигналу награды, без эксперта.

gnarl_experiments.ipynb — ноутбук с экспериментами (архитектура, BFS/DFS/
                          Bellman-Ford/MST-Prim через BC — аналог Table 2;
                          множественные решения через temperature sampling —
                          аналог Figure 2; MVC и TSP через BC+PPO — аналоги
                          Table 3 и Table 4).
```

## Проверенная корректность (сделано во время разработки)

- BFS: 200/200 корректных решений от экспертной политики на случайных графах.
- DFS: 198/200 (Algorithm 8 переведён дословно; направленный граф — тонкий
  момент, который сначала был реализован неверно через ancestor/cross-edge
  эвристику).
- Bellman-Ford, MST-Prim: 30/30.
- TSP (Held-Karp): точно совпадает с перебором на графах до 8 вершин.
- MVC (ILP): точное решение, проверено на графах Barabási–Albert.
- BC и PPO: обучение запускается и снижает loss / увеличивает награду
  (проверено на BFS и MVC).

## Осознанные упрощения (описаны в коде и в ноутбуке)

- `pred`-указатели закодированы как бинарный edge-признак, а не как
  CLRS-30 pointer-тип.
- TSP-тур строится дозаписью в конец (append), а не через pred-трюк
  "вставка в голову" из Algorithm 5 — эквивалентный по MDP-семантике выбор,
  прямо допускаемый Appendix C.2 статьи.
- В Algorithm 9 (BFS) строка `PhaseTwoPolicy` с условием `reach_j = 1`
  прочитана как `reach_j = 0` (непосещённые соседи) по аналогии с DFS —
  похоже на опечатку/ошибку OCR при выгрузке текста.
- Concorde и специализированный ILP-решатель MVC из статьи заменены на
  Held-Karp (точный DP) и PuLP/CBC соответственно — оба точны для
  используемых в ноутбуке размеров.
- Robust Graph Construction (Section 5.4) не реализован — там пространство
  действий про добавление рёбер, а не выбор узлов, что требует отдельной
  MDP-обвязки; остальные пять сред реализованы полностью.

## Запуск

```bash
pip install jax optax networkx matplotlib pandas pulp
jupyter nbconvert --to notebook --execute --inplace gnarl_experiments.ipynb
```

Ноутбук использует уменьшенные размеры графов и число эпох/апдейтов
(по сравнению со статьёй) — чтобы быстро проверить, что весь конвейер
работает целиком, а не для воспроизведения абсолютных чисел из статьи.
