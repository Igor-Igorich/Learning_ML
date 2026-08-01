
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def sigmoid(z: np.ndarray) -> np.ndarray:
    '''
    Численно устойчивая векторизованная функция Сигмоиды.
    Предотвращает Overflow float64 при больших отрицательных z.
    
    Args:
        z: Входной вектор или матрица скалярных произведений (w^T * x + b).
        
    Returns: 
        np.ndarray: Значения сигмоиды в диапазоне (0, 1).
    '''
    
    # Приводим к типу float64 для предотвращения проблем с целыми числами
    z = np.asarray(z, dtype=np.float64)
    
    # Создаем массив-результат той же формы
    res = np.zeros_like(z)
    
    # Создаем булеву маску для разделения неотрицательных и отрицательных элементов
    mask_positive = (z >= 0)
    mask_negative = ~mask_positive
    
    # 1. Для z >= 0 используем классическую формулу: 1 / (1 + exp(-z))
    # В этом случае -z <= 0, значит exp(-z) <= 1 (переполнение невозможно)
    res[mask_positive] = 1.0 / (1.0 + np.exp(-z[mask_positive]))
    
    # 2. Для z < 0 используем эквивалентную формулу: exp(z) / (1 + exp(z))
    # В этом случае z < 0, значит exp(z) < 1 (переполнение невозможно)
    exp_z_neg = np.exp(z[mask_negative])
    res[mask_negative] = exp_z_neg / (1.0 + exp_z_neg)
    
    return res

def compute_log_loss(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    eps: float = 1e-15
) -> float:
    
    '''
    Вычисление функции потерь LogLoss (Binary Cross-Entropy) с защитой от NaN.
    
    Args:
        y_true : Вектор истинных бинарных меток (0 или 1).
        y_pred_prob : Вектор предсказанных вероятностей в диапазоне [0, 1].
        eps : Минимальный сдвиг для клиппинга вероятностей.
    
    Returns:
        float: Значение ошибки LogLoss.
    '''
    # Приводим к типу float64 для предотвращения проблем с целыми числами
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float64)
    
    # Защита от log(0): отсекаем вероятности по границам [eps, 1 - eps]
    # Значения < 1e-15 станут 1e-15, значения > (1 - 1e-15) станут (1 - 1e-15)
    p_clipped = np.clip(y_pred_prob, eps, 1.0 - eps)
    
    # Поэлементное вычисление бинарной кросс-энтропии
    loss_elements = y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)
    
    # Итоговый LogLoss — это усредненный минус-логарифм правдоподобия
    log_loss = -np.mean(loss_elements)
    
    return float(log_loss)


def run_tests() -> None:
    '''
    Функция запуска набора тестов и проверки устойчивости ядра.
    '''
    logger.info('========== ТЕСТ 1: Устойчивость функции Сигмоиды ==========')
    extreme_z = np.array([-1000.0, -710.0, 0.0, 710.0, 1000.0])
    sigmoid_results = sigmoid(extreme_z)

    for z_val, sig_val in zip(extreme_z, sigmoid_results):
        logger.info(f'z = {z_val:6.1f}  -->  sigmoid(z) = {sig_val}')

    logger.info('=' * 59 + '\n')

    logger.info('=========== ТЕСТ 2: Оценка LogLoss на синтетике ===========')
    y_true = np.array([1, 0, 1, 0, 1])

    # 1. Идеальные предсказания
    p_ideal = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    loss_ideal = compute_log_loss(y_true, p_ideal)
    
    logger.info(f'1. Идеальное предсказание:    LogLoss = {loss_ideal:.15f}')

    # 2. Зеркально ошибочные предсказания
    p_wrong = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    loss_wrong = compute_log_loss(y_true, p_wrong)
    
    logger.info(f'2. Абсолютно ошибочное:       LogLoss = {loss_wrong:.6f}')

    # 3. Случайный базовый уровень (Random Baseline)
    p_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    loss_random = compute_log_loss(y_true, p_random)
    
    logger.info(f'3. Случайное (все p = 0.5):   LogLoss = {loss_random:.6f}')
    
    logger.info('=' * 59 + '\n')


def main() -> None:
    '''
    Точка входа.
    '''
    logger.info('Запуск проверки математического ядра логистической регрессии...\n')
    run_tests()
    logger.info('Все тесты успешно пройдены!')


if __name__ == '__main__':
    main()