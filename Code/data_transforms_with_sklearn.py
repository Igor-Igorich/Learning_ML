import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler, MaxAbsScaler, Normalizer, OrdinalEncoder, TargetEncoder, KBinsDiscretizer, Binarizer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union, FeatureUnion
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Глобально включаем вывод в формате Pandas
set_config(transform_output="pandas")

'''
# Профессиональный шаблон: Использование make_pipeline
pipeline = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Обращение к шагам внутри пайплайна:
# pipeline.named_steps['standardscaler']
'''

'''
# Исходный датасет
df = pd.DataFrame({
    'age': [25, 45, np.nan, 35, 50],
    'income': [50000, 80000, 120000, 65000, np.nan],
    'city': ['Moscow', 'Kazan', 'Moscow', np.nan, 'Kazan'],
    'is_subscribed': [True, False, True, False, True],
    'target': [0, 1, 1, 0, 1]
})

X = df.drop(columns=['target'])
y = df['target']
'''

# Явное использование ColumnTransformer
'''
# 1. Создаем трансформер с явными именами ('num' и 'cat')
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num', 
            make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), 
            ['age', 'income']
        ),
        (
            'cat', 
            OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
            ['city']
        )
    ],
    remainder='passthrough'  # Сохраняем столбец 'is_subscribed' без изменений
)

# Обучаем и преобразуем данные
X_transformed = preprocessor.fit_transform(X)
print(X_transformed)
'''

# Сокращенная форма make_column_transformer + make_column_selector (Production-Ready)
'''
# Создаем пайплайны обработки типов данных
num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    RobustScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
)

# Используем make_column_transformer и динамические селекторы
preprocessor = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=[object, 'category'])),
    remainder='passthrough'
)

# Готовый пайплайн вместе с моделью
full_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression()
)

# Обучение всей цепочки одной командой!
full_pipeline.fit(X, y)


# Извлечение названий признаков (get_feature_names_out)


# Предобрабатываем данные
preprocessor.fit(X)

# Получаем новые имена колонок
feature_names = preprocessor.get_feature_names_out()
print("Имена выходных признаков:\n", feature_names)
'''


'''
# Набор текстовых данных
texts = [
    "Scikit-learn documentation is helpful",
    "Machine learning in Python with pipelines",
    "FeatureUnion is great for parallel feature extraction"
]
labels = [1, 0, 1]

# 1. TF-IDF по словам (1-2 слова)
word_tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')

# 2. TF-IDF по символам (3-5 символов) - ловит опечатки и корни слов
char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char')

# 3. Объединяем их параллельно через make_union
union = make_union(word_tfidf, char_tfidf)

# 4. Собираем итоговый пайплайн
model = make_pipeline(
    union,
    LogisticRegression()
)

model.fit(texts, labels)

# Посмотрим на общее число сформированных признаков:
X_features = union.transform(texts)
print(f"Форма матрицы признаков после FeatureUnion: {X_features.shape}")
'''
'''
X, y = load_iris(return_X_y=True)

# Обязательное масштабирование перед PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создаем FeatureUnion с явными именами
combined_features = FeatureUnion(
    transformer_list=[
        ('pca', PCA(n_components=2)),             # Проекция на 2 главные компоненты
        ('select_best', SelectKBest(f_classif, k=2)) # 2 лучших исходных признака по ANOVA
    ],
    transformer_weights={
        'pca': 1.0,          # Оставляем PCA без изменений
        'select_best': 2.0   # Домножаем значения выбранных признаков на 2 (повышаем приоритет)
    }
)

X_features = combined_features.fit_transform(X_scaled)
print(f"Исходное число признаков: {X.shape[1]}")
print(f"Число признаков после FeatureUnion: {X_features.shape[1]}") # 2 + 2 = 4
'''

'''
# Кастомный трансформер, считающий длину текста и количество слов
class TextStatsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Принимает список строк X, возвращает двумерный массив N x 2
        stats = []
        for text in X:
            length = len(text)
            word_count = len(text.split())
            stats.append([length, word_count])
        return np.array(stats)

texts = ["Hello world!", "Scikit-Learn FeatureUnion guide with examples."]

# Параллельное объединение TF-IDF и ручных текстовых метрик
text_processing_union = make_union(
    TfidfVectorizer(),
    TextStatsExtractor()
)

X_out = text_processing_union.fit_transform(texts)
# Выходная матрица содержит веса слов + 2 числовых столбца метрик в конце
'''


'''
# 1. Генерируем датасет: цена имеет экспоненциальное (асимметричное) распределение
np.random.seed(42)
X = np.random.randn(200, 4)
# y = exp(X_0 + X_1) + шум
y = np.exp(X[:, 0] + X[:, 1] + 2) + np.random.gamma(shape=2, scale=1, size=200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



### Ручной путь (опасный и неудобный)
# y_train_log = np.log1p(y_train)
# model.fit(X_train, y_train_log)
#
### На этапе предсказания ЛЕГКО ЗАБЫТЬ сделать обратный exp!
# preds_log = model.predict(X_test)
# preds = np.expm1(preds_log) # Если забыть — метрики сломаются


# 2. Оборачиваем Ridge регрессию с log1p / expm1
regressor = TransformedTargetRegressor(
    regressor=Ridge(alpha=1.0),
    func=np.log1p,         # Применяется к y_train при fit()
    inverse_func=np.expm1  # Применяется к предсказаниям при predict()
)

# 3. Обучение и предсказание
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Предсказания y_pred СРАЗУ возвращаются в исходных единицах цены!
print("MAE:", mean_absolute_error(y_test, y_pred))


### Использование обучаемого трансформера (PowerTransformer / QuantileTransformer)

# Использование PowerTransformer (Yeo-Johnson) для стабилизации дисперсии y
tt_regressor = TransformedTargetRegressor(
    regressor=GradientBoostingRegressor(random_state=42),
    transformer=PowerTransformer(method='yeo-johnson')
)

# PowerTransformer автоматически вызовет fit_transform на y_train 
# и inverse_transform при вызове predict!
tt_regressor.fit(X_train, y_train)
y_pred = tt_regressor.predict(X_test)
'''
'''
### Использование в Pipeline и GridSearchCV (Production-Ready)

# Предобработка признаков X
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1]),
        ('cat', OneHotEncoder(), [2, 3])
    ]
)

# Полный pipeline обработки X и создания модели
full_model = make_pipeline(
    preprocessor,
    Lasso()
)

# Оборачиваем весь pipeline в TransformedTargetRegressor для обработки y
final_estimator = TransformedTargetRegressor(
    regressor=full_model,
    transformer=QuantileTransformer(output_distribution='normal', random_state=42)
)

# Настройка сетки гиперпараметров
# Обратить внимание на префикс 'regressor__' для доступа к внутреннему полному пайплайну
param_grid = {
    'regressor__lasso__alpha': [0.01, 0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    estimator=final_estimator,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

# Обучение с правильной оценкой качества на исходном масштабе y
grid_search.fit(X_train, y_train)
print("Лучшие параметры:", grid_search.best_params_)
'''

'''
# Создадим датасет: возраст (без выбросов) и доход (с сильным выбросом в 1,000,000)
df = pd.DataFrame({
    'age': [20, 25, 30, 35, 40],
    'income': [30000, 45000, 50000, 60000, 1000000] # 1,000,000 — аномальный выброс
})

print(f"Initial DataFrame:\n",df)
print()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Результат: среднее (mean_) близко к 0, стандартное отклонение (scale_) равно 1
print(f"DataFrame after Standard Scaler:\n", df_scaled)
print()

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
df_minmax = min_max_scaler.fit_transform(df)

# 'income': значение 1,000,000 станет 1.0, а остальные (30k-60k) сожмутся в интервал [0.0, 0.03]
print(f"DataFrame after MinMax Scaler:\n", df_minmax)
print()

maxabs_scaler = MaxAbsScaler()
df_maxabs = maxabs_scaler.fit_transform(df)

print(f"DataFrame after MaxAbs Scaler:\n", df_maxabs)
print()

robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df)

print(f"DataFrame after Robust Scaler:\n", df_robust)
print()

# Используем Yeo-Johnson (поддерживает нули и отрицательные значения)
power_scaler = PowerTransformer(method='yeo-johnson', standardize=True)
df_power = power_scaler.fit_transform(df)

print(f"DataFrame after Power Transformer with (method='yeo-johnson', standardize=True):\n", df_power)
print()

# output_distribution='uniform': Преобразует данные в равномерное распределение [0, 1].
# output_distribution='normal': Преобразует данные в стандартное нормальное распределение.

quantile_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
df_quantile = quantile_scaler.fit_transform(df)

print(f"DataFrame after Quantile Transformer with output_distribution='normal':\n", df_quantile)
print()



### Важная путаница:
# Не путать MinMaxScaler (который масштабирует столбцы)
# с Normalizer (который нормализует строки).

# Normalizer измеряет векторы-строки x и масштабирует их так,
# чтобы их математическая норма была равна 1 (единичный вектор):
#   L1-норма (norm='l1'): Сумма абсолютных значений элементов строки равна 1 (sum{|x_i|} = 1)
#   L2-норма (norm='l2'): Евклидова длина вектора равна 1 (sqrt{sum x_i^2} = 1)
#   Max-норма (norm='max'): Максимальное значение строки становится равным 1

# Нормализация каждой строки отдельно (по L2-норме)
normalizer = Normalizer(norm='l2')
df_normalized = normalizer.fit_transform(df)

print(f"DataFrame after Normalizer with norm='l2' (sqrt{{sum x_i^2}} = 1):\n", df_normalized)
print()
'''

'''
df = pd.DataFrame({
    'city': ['Moscow', 'Kazan', 'Moscow', 'Saint P', np.nan], # Номинальный признак
    'size': ['S', 'M', 'L', 'S', 'M'],                       # Порядковый признак
    'zip_code': ['101000', '420000', '101000', '190000', '420000'], # Выс. мощность (High-cardinality)
    'target': [0, 1, 0, 1, 1]                                 # Целевая переменная
})

X = df.drop(columns=['target'])
y = df['target']

print(f"Initial X:\n", X)
print()

### OneHotEncoder:

# handle_unknown='ignore' (Must-have в продакшене):
#   Если на тесте придет неизвестная категория, кодировщик не выдаст ошибку,
#   а заполнит строку нулями.
# drop='first' / 'if_binary':
#   Удаляет один столбец для предотвращения идеальной мультиколлинеарности (дамми-ловушки)
#   в линейных моделях.
# sparse_output=False:
#   Принудительно возвращает плотную матрицу
#   или DataFrame вместо scipy.sparse.

# Внимание при комбинации drop='first' и handle_unknown='ignore':
#   В Scikit-Learn с версии 1.1 можно использовать эти параметры вместе
#   только при handle_unknown='infrequent_if_exist'. Для
#   обычного handle_unknown='ignore' параметр drop не задают
#   или используют drop=None.

ohe = OneHotEncoder(
    drop="if_binary", # Если категории всего 2, сделает 1 столбец вместо 2
    handle_unknown="ignore",
    sparse_output=False
)

X_city_ohe = ohe.fit_transform(X[["city"]])
print(f"X['city'] after OneHot Encoder:\n", X_city_ohe)
print()

# В OneHotEncoder можно автоматически объединять редкие категории
# (например, встречающиеся менее 1% раз) в одну общую категорию "infrequent".
#
# ohe = OneHotEncoder(min_frequency=0.01, handle_unknown='ignore')


### OrdinalEncoder

# categories:
#   Позволяет вручную задать явный порядок значений.
# handle_unknown='use_encoded_value' + unknown_value=-1:
#   Безопасная обработка невидимых ранее категорий.
# encoded_missing_value=-2:
#   Позволяет явно закодировать пропуски NaN числом, не вызывая импутатор.

# Явно задаем правильный порядок для размера одежды
size_order = ['S', 'M', 'L']

ordinal_enc = OrdinalEncoder(
    categories=[size_order],
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-2
)

X_size_encoded = ordinal_enc.fit_transform(X[['size']])
print(f"X['size'] after Ordinal Encoder:\n", X_size_encoded)
print()


### TargetEncoder

# TargetEncoder в Scikit-Learn использует внутренний cross-fitting
# во время fit_transform(). Он автоматически разбивает обучающую
# выборку на фолды и вычисляет целевые значения для фолда по оставшимся фолдам.

target_enc = TargetEncoder(
    smooth="auto",      # Автоматическая регуляризация сглаживания
    cv=2                # Количество фолдов для кросс-фиттинга при fit_transform
)

X_zip_encoded = target_enc.fit_transform(X[['zip_code']], y)

print(f"X['zip_code'] after Target Encoder:\n", X_zip_encoded)
print()
'''

'''
df = pd.DataFrame({
    'income': [15000, 18000, 22000, 25000, 40000, 50000, 60000, 80000, 120000, 1000000]
})

est_quantile = KBinsDiscretizer(
    n_bins=4,
    encode='ordinal',
    strategy='quantile'
)

df_quantile = est_quantile.fit_transform(df[['income']])
print(df_quantile)

print("\nГраницы корзин (bin_edges_):")
print(est_quantile.bin_edges_)

est_uniform = KBinsDiscretizer(
    n_bins=3, 
    encode='onehot-dense', 
    strategy='uniform'
)

df_onehot = est_uniform.fit_transform(df[['income']])
print(df_onehot)

data = pd.DataFrame({
    'age': [22, 55, 33, 45, 61, 19],
    'salary': [30000, 90000, 50000, 120000, 85000, 25000],
    'gender': ['M', 'F', 'F', 'M', 'F', 'M'],
    'target': [0, 1, 0, 1, 1, 0]
})

X = data.drop(columns=['target'])
y = data['target']

# Настраиваем обработку:
# - 'age' дискретизируем на 3 корзины
# - 'salary' масштабируем через StandardScaler
# - 'gender' кодируем через OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('age_bins', KBinsDiscretizer(n_bins=3, encode='onehot', strategy='quantile'), ['age']),
        ('salary_scale', StandardScaler(), ['salary']),
        ('gender_ohe', OneHotEncoder(drop='if_binary'), ['gender'])
    ]
)

# Собираем итоговый пайплайн
pipeline = make_pipeline(preprocessor, LogisticRegression())
pipeline.fit(X, y)

# Все значения > 50000 станут 1, остальные 0
binarizer = Binarizer(threshold=50000)
df_binarized = binarizer.fit_transform(df[['income']])

print(df_binarized)

'''

# Создаем трансформер для логарифмирования np.log1p (log(1 + x))
log_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
    validate=False
)

df = pd.DataFrame({'income': [10000, 50000, 150000]})

# Логарифмируем
df_log = log_transformer.fit_transform(df)
print("После логарифмирования:\n", df_log)

# Восстанавливаем исходные значения
df_original = log_transformer.inverse_transform(df_log)

def extract_date_features(X):
    """Принимает DataFrame с колонкой 'date' и возвращает новые признаки."""
    
    X_out = X.copy()
    dates = pd.to_datetime(X_out['date'])
    
    X_out['year'] = dates.dt.year
    X_out['month'] = dates.dt.month
    X_out['dayofweek'] = dates.dt.dayofweek
    
    return X_out.drop(columns=['date'])

# Создаем трансформер (validate=False важен для работы с Pandas DataFrame)
date_transformer = FunctionTransformer(extract_date_features, validate=False)

df_dates = pd.DataFrame({'date': ['2026-01-15', '2026-08-03']})
df_features = date_transformer.fit_transform(df_dates)
print(df_features)


### Архитектура класса в Scikit-Learn:
# Для полной интеграции с экосистемой Scikit-Learn класс должен наследоваться
# от двух базовых классов:

# TransformerMixin:

#   Автоматически добавляет метод fit_transform(X, y) на основе ваших методов fit() и transform()
#   Позволяет использовать метод set_output(transform="pandas").

# BaseEstimator:

#   Автоматически добавляет методы get_params() и set_params()
#   Важное правило: В методе __init__ все параметры должны приниматься
#     в явном виде с дефолтными значениями и просто присваиваться через
#     self.param = param без вычислений и изменений! Это критично для
#     корректной работы GridSearchCV.

### Требования к написанию методов:
#   __init__(self, ...):
#       Объявление гиперпараметров трансформера.

#   fit(self, X, y=None):
#       Вычисляет внутренние параметры на основе X (и y, если нужно).
#         Вычисленные параметры сохраняются в атрибуты класса, заканчивающиеся на
#         нижнее подчеркивание _ (например, self.mean_, self.quantile_bounds_).
#         Это соглашение API Scikit-Learn.
#       Обязательно возвращает self!

#   transform(self, X):
#       Применяет сохраненные параметры к X и возвращает преобразованные данные.

#   get_feature_names_out(self, input_features=None) (Опционально):
#       Возвращает имена столбцов на выходе.


'''
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Трансформер для ограничения выбросов по заданным квантилям.
    """
    
    def __init__(
        self,
        lower_quantile: float=0.05,
        upper_quantile: float=0.95
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        # Вычисляем параметры на TRAIN и сохраняем в атрибуты с _ на конце
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        return X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        
        return self.feature_names_in_
    
# Демонстрация использования
X_train = pd.DataFrame({'price': [10, 100, 105, 110, 115, 120, 1000]}) # 10 и 1000 - выбросы
X_test  = pd.DataFrame({'price': [5, 112, 2000]})

clipper = OutlierClipper(lower_quantile=0.10, upper_quantile=0.90)

# Обучаем на train
X_train_clipped = clipper.fit_transform(X_train)
print("Обученный Train:\n", X_train_clipped)

# На тесте используем границы, посчитанные на train
X_test_clipped = clipper.transform(X_test)
print("\nПреобразованный Test:\n", X_test_clipped)


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Генерирует Debt-to-Income ratio и объединяет редкие категории работы.
    """
    def __init__(self, min_category_freq: int = 2):
        self.min_category_freq = min_category_freq

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        # Находим частые категории работы на Train
        job_counts = X_df['job_type'].value_counts()
        self.frequent_jobs_ = set(job_counts[job_counts >= self.min_category_freq].index)
        
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # 1. Расчет нового числового признака
        X_df['dti_ratio'] = X_df['total_debt'] / (X_df['annual_income'] + 1e-5)
        
        # 2. Замена редких категорий на 'Other' с помощью сохраненного множества frequent_jobs_
        X_df['job_type'] = X_df['job_type'].apply(
            lambda job: job if job in self.frequent_jobs_ else 'Other'
        )
        
        return X_df

# Тестовые данные
df_credit = pd.DataFrame({
    'annual_income': [50000, 100000, 30000, 80000],
    'total_debt': [10000, 15000, 20000, 5000],
    'job_type': ['Developer', 'Developer', 'Designer', 'Astronaut']
})

fe_transformer = CreditFeatureEngineer(min_category_freq=2)
df_transformed = fe_transformer.fit_transform(df_credit)
print(df_transformed)
'''

