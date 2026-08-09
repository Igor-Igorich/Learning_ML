import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

"""
x = np.arange(7)
y_1 = np.array([3, 1, 2, 4, 5, 3, 4])
y_2 = np.array([5, 4, 5, 3, 4, 2, 5])


plt.figure(figsize=(9, 6))
plt.plot(x, y_1, label="Петя", color="red")
plt.plot(x, y_2, label="Вася", color="green")

plt.title("Зависимость количества чашек\nчая в день от дня недели", fontsize=17)

plt.xlabel("День недели", fontsize=15)
plt.ylabel("Количество чашек чая в день", fontsize=15)

plt.xticks(
    ticks=np.arange(7),
    labels=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    fontsize=12,
    rotation=30,
)
plt.yticks(ticks=np.arange(1, 6))

plt.legend(
    title="Люди",
    title_fontsize=14,
    fontsize=11,
    ncol=2,
    loc="lower right",
)

plt.grid()
plt.show()
"""

"""
x = np.random.normal(size=10)
y = np.random.normal(size=10)

plt.figure(figsize=(9, 6))
plt.scatter(
    x,
    y,
    c=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
)

plt.title("Цвета задали с помощью массива цветовых кодов", fontsize=17)

plt.show()
"""

"""
plt.figure(figsize=(9, 6))
x = np.random.normal(size=100)
y = np.random.normal(size=100)
plt.scatter(x, y, c=y > 0)

plt.title("Цвета задали с помощью массива чисел", fontsize=17)

plt.show()
"""

"""
x = np.linspace(-5, 5, 100)
y = np.sin(x)

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(y)
plt.title("Странный график функции sin(x)", fontsize=17)
plt.xlabel(
    "Значения на этой оси заданы по умолчанию,\nсоответствуют номеру точки",
    fontsize=15,
)
plt.ylabel("Значение функции sin(x)", fontsize=15)
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title("График функции sin(x)", fontsize=17)
plt.xlabel(
    "Значения на этой оси заданы вручную,\nсоответствуют аргументу x",
    fontsize=15,
)
plt.ylabel("Значение функции sin(x)", fontsize=15)
plt.grid()

plt.show()
"""

"""
x = np.linspace(-5, 5, 100)
y = np.sin(x)

plt.figure(figsize=(11, 8))

plt.title("Различные значения параметров ls и lw", fontsize=17)

plt.plot(x, y, linestyle="-", linewidth=1, label="ls = '-',   lw = 1")

plt.plot(x, y + 1, linestyle="--", linewidth=2, label="ls = '--',  lw = 2")

plt.plot(x, y + 2, linestyle="-.", linewidth=4, label="ls = '-.',  lw = 4")

plt.plot(x, y + 3, linestyle=":", linewidth=6, label="ls = ':',   lw = 6")

plt.legend(
    title="аргументы ls и lw",
    fontsize=12,
    title_fontsize=13,
    loc="lower right",
    shadow=True,
)

plt.grid(linestyle="-.", linewidth=2)
plt.show()
"""

"""
x = np.arange(7)
y1 = np.array([3, 1, 2, 4, 5, 3, 4])
y2 = np.array([5, 4, 5, 3, 4, 2, 5])

# marker:
# - '.' для точки
# - 'o' для круга
# - '^' для треугольника
# - 's' для квадрата

plt.figure(figsize=(9, 6))
plt.plot(
    x,
    y1,
    label="Петя",
    marker="^",
    markersize=20,
    color="blue",
    markerfacecolor="orange",
)

plt.plot(
    x,
    y2,
    label="Вася",
    marker="s",
    markersize=10,
    color="orange",
    markerfacecolor="yellow",
    markeredgecolor="green",
)

plt.title("Зависимость количества чашек\nчая в день от дня недели", fontsize=17)

plt.xlabel("День недели", fontsize=15)
plt.ylabel("Количество чашек чая в день", fontsize=15)

plt.xticks(
    np.arange(7),
    labels=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    fontsize=12,
    rotation=30,
)
plt.yticks(np.arange(1, 6))

plt.legend(
    title="Люди", title_fontsize=13, fontsize=12, ncol=2, loc="lower right"
)

plt.grid()
plt.show()
"""

"""
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.scatter(x, y, s=25, marker="^", color="green")
plt.title("Параметры для всех одни и те же", fontsize=17)

plt.subplot(1, 2, 2)
plt.scatter(
    x, y, s=[5 if y[i] < 0 else 50 for i in range(1000)], marker="s", c=(x < 0)
)
plt.title("Для каждой точки своё значение цвета и размера", fontsize=17)

plt.show()
"""

"""
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig.set(facecolor="green")
ax.set(facecolor="red")

plt.show()
"""

"""
x = np.arange(7)
y1 = np.array([3, 1, 2, 4, 5, 3, 4])
y2 = np.array([5, 4, 5, 3, 4, 2, 5])

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)

ax.plot(x, y1, label="Петя")
ax.plot(x, y2, label="Вася")

ax.set_title(
    "Зависимость количества чашек\nчая в день от дня недели", fontsize=17
)

ax.set_xlabel("День недели", fontsize=15)
ax.set_ylabel("Количество чашек чая в день", fontsize=15)

ax.set_yticks(np.arange(1, 6))

ax.legend(
    title="Люди", title_fontsize=13, fontsize=12, ncol=2, loc="lower right"
)

ax.grid()
plt.show()
"""

"""
fig = plt.figure(figsize=(9, 6))
fig.suptitle("Заголовок области Figure", fontsize=20)

ax = fig.add_subplot(1, 1, 1)
ax.set_title("Заголовок области Axis", fontsize=15)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
"""

"""
fig = plt.figure(figsize=(10, 8))
fig.suptitle("Заголовок области Figure")

ax_1 = fig.add_subplot(2, 2, 2)
ax_2 = fig.add_subplot(2, 2, 3)
ax_3 = fig.add_subplot(3, 3, 5)
ax_4 = fig.add_subplot(3, 3, 9)

ax_1.set(title="ax_1", xticks=[], yticks=[])
ax_2.set(title="ax_2", xticks=[], yticks=[])
ax_3.set(title="ax_3", xticks=[], yticks=[])
ax_4.set(title="ax_4", xticks=[], yticks=[])

plt.show()
"""

"""
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].set(title="axes[0, 0]", xticks=[], yticks=[])
axes[0, 1].set(title="axes[0, 1]", xticks=[], yticks=[])
axes[1, 0].set(title="axes[1, 0]", xticks=[], yticks=[])
axes[1, 1].set(title="axes[1, 1]", xticks=[], yticks=[])

plt.show()
"""

"""
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

axes[0, 0].set(title="axes[0, 0]")
axes[0, 1].set(title="axes[0, 1]")
axes[1, 0].set(title="axes[1, 0]")
axes[1, 1].set(title="axes[1, 1]")

plt.show()
"""

"""
from io import BytesIO

import requests

url = "https://cs13.pikabu.ru/avatars/3128/x3128007-1508104989.png"

response = requests.get(url)
response.raise_for_status()

image = plt.imread(BytesIO(response.content))

print("Было:")
fig, axes = plt.subplots(2, 2, figsize=(4, 4))

axes[0, 0].imshow(image)
axes[0, 1].imshow(image)
axes[1, 0].imshow(image)
axes[1, 1].imshow(image)

plt.show()


print("Стало:")
fig, axes = plt.subplots(2, 2, figsize=(4, 4))

axes[0, 0].imshow(image)
axes[0, 0].axis("off")
axes[0, 1].imshow(image)
axes[0, 1].axis("off")
axes[1, 0].imshow(image)
axes[1, 0].axis("off")
axes[1, 1].imshow(image)
axes[1, 1].axis("off")

plt.show()
"""

"""
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("аналог axes[0, 0]")

plt.subplot(2, 2, 4)
plt.title("аналог axes[1, 1]")

plt.subplot(2, 3, 4)
plt.title("аналог axes[1, 0]")

plt.subplot(3, 2, 2)
plt.title("аналог axes[0, 1]")

plt.show()
"""

"""
tips = sns.load_dataset("tips")

sns.relplot(
    x="total_bill", y="tip", hue="time", col="smoker", row="sex", data=tips
)

plt.show()
"""

"""
fmri = sns.load_dataset("fmri")

sns.relplot(
    x="timepoint",
    y="signal",
    kind="line",
    hue="region",
    height=8,
    aspect=1.5,
    data=fmri,
)

plt.show()
"""

"""
tips = sns.load_dataset("tips")
ds = pd.pivot_table(data=tips, index="day", columns="size", values="tip")

fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.heatmap(
    ds, ax=ax, yticklabels=["Четверг", "Пятница", "Суббота", "Воскресенье"]
)

ax.set_title(
    "Значение tip в зависимости от дня недели и размера группы\n", fontsize=15
)
ax.set_ylabel("День недели", fontsize=12)
ax.set_xlabel("Размер группы", fontsize=12)

plt.show()
"""


"""
np.random.seed(191)

t = np.linspace(0, 5, 200) * np.ones((7, 200)) + np.random.randn(7, 200).cumsum(
    axis=1
)
a = np.array([0.5, 1, 2.1, -1.3, -0.1, 1.9, -2.9]).reshape(-1, 1)
b = np.array([-3, -4, -10, 4, 1, -5, 6]).reshape(-1, 1)

df = pd.DataFrame((a * t + b).T, columns=list("ABCDEFG"))

# plt.figure(figsize=(11, 9))
# plt.plot(df.values)
# plt.legend(labels=df.columns, title="Признаки", title_fontsize=13, fontsize=12)
# plt.title(
#     "Случайные данные с некоторой корреляцией между признаками", fontsize=15
# )

fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.heatmap(data=df.corr(), ax=ax, annot=True)
ax.set_title("Корреляционная матрица с указанием значений", fontsize=15)

plt.show()
"""

"""
penguins = sns.load_dataset("penguins")

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].set_title("Распределение длины крыльев пингвинов (мм)", fontsize=15)
sns.histplot(
    penguins.flipper_length_mm,
    bins=30,
    ax=ax[0, 0],
    color="green",
    edgecolor="darkgreen",
)

ax[0, 1].set_title("Распределение длины крыльев пингвинов (мм)", fontsize=15)
sns.histplot(
    y=penguins.flipper_length_mm,
    bins=30,
    ax=ax[0, 1],
    color="green",
    edgecolor="darkgreen",
)

ax[1, 0].set_title(
    "Распределение длины крыльев пингвинов (мм)\nв зависимости от вида",
    fontsize=15,
)
sns.histplot(
    data=penguins, x="flipper_length_mm", hue="species", bins=30, ax=ax[1, 0]
)

ax[1, 1].set_title("Распределение и функция плотности", fontsize=15)
sns.histplot(
    data=penguins,
    x="flipper_length_mm",
    hue="species",
    kde=True,
    bins=30,
    ax=ax[1, 1],
)

plt.show()
"""

"""
penguins = sns.load_dataset("penguins")

sns.displot(
    data=penguins,
    x="flipper_length_mm",
    hue="species",
    kde=True,
    bins=30,
    height=6,
    aspect=1.5,
)

sns.displot(
    data=penguins,
    x="flipper_length_mm",
    hue="species",
    kind="kde",
    height=6,
    aspect=1.5,
)

sns.displot(
    data=penguins,
    x="flipper_length_mm",
    hue="species",
    kind="ecdf",
    height=6,
    aspect=1.5,
)

sns.displot(
    data=penguins, x="flipper_length_mm", hue="species", col="sex", kind="ecdf"
)

plt.show()
"""

"""
penguins = sns.load_dataset("penguins")

sns.pairplot(penguins)
sns.pairplot(penguins, hue="species")
plt.show()
"""


# %config InlineBackend.figure_format = 'retina'

# # sns.set(style='whitegrid', palette='deep')
# # sns.set(style='darkgrid', palette='rocket')
# sns.set(style='darkgrid', palette='deep')

# plt.rcParams['figure.figsize'] = 8, 5
# plt.rcParams['font.size'] = 12
# plt.rcParams['savefig.format'] = 'pdf'
