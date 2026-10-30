# AD7031 Optimization Under Uncertainty

<p align="center">
  <img src="assets/qrcode.png" alt="Repository QR code" width="220"/>
</p>

Course materials for **AD7031 불확실성하 의사결정 (Optimization Under Uncertainty)** — lecture notes, interactive
dashboards, and Colab-ready notebooks, updated weekly as the course progresses.

📄 [Syllabus (PDF)](syllabus/%5B%EA%B0%95%EC%9D%98%EA%B3%84%ED%9A%8D%EC%84%9C%5D%EB%B6%88%ED%99%95%EC%8B%A4%EC%84%B1%ED%95%98%20%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%28AD7031%29_26_2.pdf)
🎮 [Optimization Games](https://hankpark0706.github.io/ip-games/) — browser-based games for the classic models (Markowitz portfolio, knapsack, assignment, shortest path, transportation, min-cost flow). Nothing to install.

---

## Weekly materials

| Week | Date | Topic | Lecture note | Coding session | Video | Misc |
|---|---|---|:---:|:---:|:---:|---|
| 1 | 8/29 | Intro: optimization under uncertainty (OUU) is everywhere | [week01](lecture-notes/OUU_week01_slide.pdf) | |  | [읽을거리](https://hankpark0706.github.io/AD7031/week01_reading.html) ([PDF](lecture-notes/OUU_week01_reading.pdf)) |
| 2 | 9/5 | Optimization theory review — convex optimization; probability theory review | [week02](lecture-notes/OUU_week02_note.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hankpark0706/AD7031/blob/main/notebooks/week02_linear_program.ipynb) *linear program* | [Optimization: A Bootcamp for Machine Learning, Inverse Problems, and Control](https://www.youtube.com/watch?v=lPBPbGmw1_4) | |
| 3 | 9/12 | Linear and quadratic programs; convexity-preserving operations; probability review (start) | [week02](lecture-notes/OUU_week02_note.pdf) · [week04](lecture-notes/OUU_week04_note.pdf) | | [George Dantzig, The Accidental Maths Genius (Good Will Hunting in Real Life)](https://www.youtube.com/watch?v=xynFFqcSZDY) | **[HW#1](homework/hw01.pdf) out** (due 9/26) |
| 4 | 9/19 | Probability review — events and conditioning, scenarios, expectation; Markowitz portfolio | [week04](lecture-notes/OUU_week04_note.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hankpark0706/AD7031/blob/main/notebooks/week03_markowitz.ipynb) *markowitz* | | 1. [Seeing Theory](https://seeing-theory.brown.edu/probability-distributions/index.html) — 1D distributions<br>2. [Bivariate Gaussian](https://mtalebi.github.io/Bivariate-Gaussian-Distribution/) — 2D distributions |
| 5 | 9/26 | *No class* | | | | **[HW#1 solution](homework/hw01-sol.pdf)** |
| 6 | 10/3 | Law of large numbers; duality; conic programming | [week06](lecture-notes/OUU_week06_note.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hankpark0706/AD7031/blob/main/notebooks/week06_markowitz_max_return.ipynb) *markowitz, max return* | 1. [Newton's Method in Optimization](https://www.youtube.com/watch?v=W7S94pq5Xuo)<br>2. [Dual problem](https://www.youtube.com/watch?v=uh1Dk68cfWs) | |
| 7 | 10/10 | **Project topic pitch** — discussion of draft final-project ideas | 1. [Sign-up sheet](https://docs.google.com/spreadsheets/d/1tJ84bbIaPR2J5Y_F0ocXhTwKVYQEOGXXVd-yIYQU66Y/edit?gid=258135268#gid=258135268)<br>2. [발표참고자료](final-project/AD7031_project_template.pdf) | | | **Project topic pitch** (oral) on [Zoom](https://illinois.zoom.us/j/9545298219?pwd=RmpzSG5SYmJNUEx4SWNwQVhLL3BaZz09) |
| 8 | 10/17 | Duality; conic programming (cont.) | | | | **[HW#2](homework/hw02.pdf) out** (due 10/31) |
| 9 | 10/24 | Risk measures 1 | [week09](lecture-notes/OUU_week09_note.pdf) | | | |
| 10 | 10/31 | Chance constraints — ML is optimization under uncertainty | [OL7014 week04](lecture-notes/OL7014_week04_slide.pdf) | | | [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.11962&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) |
| 11 | 11/7 | Risk measures 2 | | | | |
| 12 | 11/14 | Two-stage models 1 | | | | **HW#3 out** (due 12/5) |
| 13 | 11/21 | Two-stage models 2 | | | | |
| 14 | 11/28 | Two-stage models 3 | | | | |
| 15 | 12/5 | Distributionally robust optimization | | | | **HW#3 due** · 기말고사 review |
| 16 | 12/12 | **Final exam** | | | | Written exam |

Lecture notes, notebooks, and demos are posted week by week — check back after each lecture.

## Assessment

The schedule above revises the one in the syllabus: the midterm and the final project are
replaced by a single **final exam in week 16 (12/12)**, a third problem set is added, and the
remaining material is spread over the rest of the semester.

|  | syllabus | revised |
|---|:---:|:---:|
| Homework | 20% (2 problem sets) | **30%** (3 problem sets) |
| Midterm | 30% | — |
| Final project | 40% | — |
| Final exam | — | **60%** |
| Attendance | 10% | **10%** |

- **HW#3** — out 11/14, due 12/5
- **Final exam** — week 16 (12/12), covering weeks 1–15
- The week 7 topic pitches count towards attendance.

## Reference

G. Hanasusanto, H. Park, *Optimization Under Uncertainty: Models, Methods, and Guarantees* (draft), 2026.
