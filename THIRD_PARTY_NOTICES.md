# Third-party notices

blackopt reuses design and algorithm work from other open-source projects.
This file reproduces their required license notices.

## Optuna

blackopt's public API and several of its algorithm implementations are **modeled
on and adapted from [Optuna](https://github.com/optuna/optuna)** — including the
study/trial/sampler model, the define-by-run and ask-and-tell interfaces
(`create_study`, `study.optimize`, `study.ask`/`tell`, `suggest_float`/`suggest_int`/
`suggest_categorical`, `StudyDirection`, `FrozenTrial`), the sampler lineup (TPE,
Random, Grid, QMC, CMA-ES, NSGA-II/III, BruteForce, PartialFixed), the pruners,
and fANOVA parameter-importance analysis. The underlying algorithms carry their
own primary citations (see the README references); this notice credits Optuna,
whose implementations and API those parts were derived from.

Optuna is licensed under the MIT License:

```
MIT License

Copyright (c) 2018 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
