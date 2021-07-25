import numpy as np
import time
import sys
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

start_time = time.time()

# Инициализация
# Разме таблицы
table_size = (3, 5)
# Размер прямоугольных полимино и их мощность
q_figure = [((2, 2), 1)]
# Размер L-полимино и их мощность
l_figure = [((3, 2), 1), ((2, 2), 1)]
# Распаковка ( мощность и размер)
q_size, q_figure_count = zip(*q_figure)
l_size, l_figure_count = zip(*l_figure)
print(f"Размеры: {q_size}, " f"Мощности: {q_figure_count}")
print(f"Размеры: {l_size}, " f"Мощности: {l_figure_count}")
dim_1, dim_2 = table_size  # Размеры доски
currentPosition = 0  # Текущая позиция на доске
dlx_table = np.zeros(dim_1 * dim_2)  # Начало матрицы возможных постановок фигур


def set_positions():
    # Определение позиций на доске
    # Каждая новая строка это новый десяток
    # Поэтому программа рассчитана на поле не больше 10X10
    # Для 100X100 необходимо выставить k + t * 100 и т.д
    positions = []
    start_position = 11
    for _t in range(dim_1):
        for k in range(dim_2):
            positions.append(start_position + k + _t * 10)
    positions = np.array(positions)
    return positions


columns = set_positions()


# Функция сдвига позиции вправо или влево
def add(dim1, _current_col, update_array, sign):
    _currentPosition = 0
    for n in range(dim1):
        _currentPosition = _current_col + n * sign
        if np.any(columns == _currentPosition):
            ind = list(columns).index(_currentPosition)
            update_array[ind] = 1
        else:
            return _currentPosition, update_array, False
    return _currentPosition, update_array, True


# Функция сдвига позиции вверх или вниз
def multi(dim2, _current_col, update_array, sign):
    _currentPosition = 0
    for q in range(dim2):
        _currentPosition = _current_col + 10 * q * sign
        if np.any(columns == _currentPosition):
            ind = list(columns).index(_currentPosition)
            update_array[ind] = 1
        else:
            return _currentPosition, update_array, False
    return _currentPosition, update_array, True


# Поиск в 2 этапах:
# 1. +-a, +-b * 10
# 2. +-a * 10, +-b
def calculate_dlx_table(table, _l_size):
    a, b = _l_size
    size = table.size
    # Проход по позициям фигуры
    for _i in range(columns.size):
        # Первый этап
        # +a, +b * 10
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        sign = 1
        current_column, UpdateArray, check_prob = add(a, current_column, UpdateArray, sign)
        if check_prob:
            current_column, UpdateArray, check_prob = multi(b, current_column, UpdateArray, sign)
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # -a, +b * 10
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = add(a, current_column, UpdateArray, sign * (-1))
        if check_prob:
            current_column, UpdateArray, check_prob = multi(b, current_column, UpdateArray, sign)
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # +a, -b * 10
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = add(a, current_column, UpdateArray, sign)
        if check_prob:
            current_column, UpdateArray, check_prob = multi(b, current_column, UpdateArray, sign * (-1))
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # -a, -b * 10
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = add(a, current_column, UpdateArray, sign * (-1))
        if check_prob:
            current_column, UpdateArray, check_prob = multi(b, current_column, UpdateArray, sign * (-1))
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # Второй этап
        # +a * 10, +b
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = multi(a, current_column, UpdateArray, sign)
        if check_prob:
            current_column, UpdateArray, check_prob = add(b, current_column, UpdateArray, sign)
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # -a * 10, +b
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = multi(a, current_column, UpdateArray, sign * (-1))
        if check_prob:
            current_column, UpdateArray, check_prob = add(b, current_column, UpdateArray, sign)
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # +a * 10, -b
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = multi(a, current_column, UpdateArray, sign)
        if check_prob:
            current_column, UpdateArray, check_prob = add(b, current_column, UpdateArray, sign * (-1))
            if check_prob:
                table = np.vstack((UpdateArray, table))
        # -a * 10, -b
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = multi(a, current_column, UpdateArray, sign * (-1))
        if check_prob:
            current_column, UpdateArray, check_prob = add(b, current_column, UpdateArray, sign * (-1))
            if check_prob:
                table = np.vstack((UpdateArray, table))
    return table


# Функция сдвига влево или вправо на 1 и сразу после этого вниз или вверх (для прямоугольных фигур)
def add_and_multi(dim1, dim2, _current_col, update_array):
    _currentPosition = 0
    for n in range(dim1):
        _currentPosition = _current_col + n
        if np.any(columns == _currentPosition):
            ind = list(columns).index(_currentPosition)
            update_array[ind] = 1
            for k in range(dim2):
                _currentPosition = _currentPosition + 10 * k
                if np.any(columns == _currentPosition):
                    ind = list(columns).index(_currentPosition)
                    update_array[ind] = 1
                else:
                    return _currentPosition, update_array, False
        else:
            return _currentPosition, update_array, False
    return _currentPosition, update_array, True


# Наоборот соответсвенно
def multi_and_add(dim1, dim2, _current_col, update_array):
    _currentPosition = 0
    for n in range(dim1):
        _currentPosition = _current_col + 10 * n
        if np.any(columns == _currentPosition):
            ind = list(columns).index(_currentPosition)
            update_array[ind] = 1
            for k in range(dim2):
                _currentPosition = _currentPosition + k
                if np.any(columns == _currentPosition):
                    ind = list(columns).index(_currentPosition)
                    update_array[ind] = 1
                else:
                    return _currentPosition, update_array, False
        else:
            return _currentPosition, update_array, False
    return _currentPosition, update_array, True


# Поиск позиций прямоугольника
# 1.+c AND +10*d
# 2.+10*c AND +d
def calculate_dlx_table_square(table, _q_size):
    _c, _d = _q_size
    size = table.size
    for _i in range(columns.size):
        # Первый этап
        # +c AND +10*d
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = add_and_multi(_c, _d, current_column, UpdateArray)
        if check_prob:
            table = np.vstack((UpdateArray, table))
        # Второй этап
        # +10*c AND +d
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        check_prob = True
        current_column = columns[_i]
        UpdateArray = np.zeros(size)
        current_column, UpdateArray, check_prob = multi_and_add(_c, _d, current_column, UpdateArray)
        if check_prob:
            table = np.vstack((UpdateArray, table))
    return table


list_of_matrix = []
# Проходимся по всем фигурам, отдаём их функциям вычисления позиций.
for number in range(len(l_figure_count)):
    new_dlx_table = calculate_dlx_table(dlx_table, l_size[number])
    for count in range(l_figure_count[number]):
        list_of_matrix.append(new_dlx_table[:-1])
for number in range(len(q_figure_count)):
    new_dlx_table = calculate_dlx_table_square(dlx_table, q_size[number])
    for count in range(q_figure_count[number]):
        list_of_matrix.append(new_dlx_table[:-1])
dlx_table = np.array(list_of_matrix)

# print(sum(l_figure_count) + sum(q_figure_count))
# Создаём матрицу смежности, добавляя n-количество столбцов (где n количество фигур).
# 1 где фигура смежна с данной позицией.
new_dlx_table = np.zeros((1, len(columns) + len(dlx_table)))
for y in range(len(dlx_table)):
    current_table = dlx_table[y]
    num_rows, num_cols = current_table.shape
    s = num_rows
    t = np.zeros((s, len(dlx_table)))
    for j in range(num_rows):
        t[j, y] = 1
    dlx_table[y] = np.hstack((current_table, t))
    new_dlx_table = np.vstack((new_dlx_table, dlx_table[y]))
dlx_table = new_dlx_table[::-1]
# print(dlx_table[:10])
num_rows, num_cols = new_dlx_table.shape
node_table = []
# Переделываем нашу матрицу смежности в лист нод
for i in range(num_rows):
    for j in range(num_cols):
        if dlx_table[i][j] == 1:
            node_table.append((i, j))


# print(node_table)
def time_passed():
    print("time elapsed: {:.2f}s".format(time.time() - start_time))


class CoverError(Exception):
    pass


A = node_table

B = {}  # Здесь будут решения
updates = {}  # Для подсчёта количества ребер графа
covered_cols = {}  # Проверка покрытости
for (r, c) in A:
    covered_cols[c] = False


def print_solution():
    if len(B) == (sum(q_figure_count) + sum(l_figure_count)):
        # Для определения всех возможных вариаций
        # print(("SOLUTION", updates))
        # for k in B:
        #    for node in B[k]:
        #        print((node[1]))
        #    print("--------------")
        print("TRUE")
        time_passed()
        sys.exit()


def choose_col():
    # Выбор того столбца вкотором меньше всего едениц
    # Ищем столбцы где всё ещё False покрытия
    cols = [c for c in covered_cols if not covered_cols[c]]
    if not cols:
        raise CoverError("all columns are covered")
    tmp = dict([(c, 0) for c in cols])
    for (r, c) in A:
        if c in cols:
            tmp[c] = tmp[c] + 1
    min_c = cols[0]
    for c in cols:
        if tmp[c] < tmp[min_c]:
            min_c = c
    return min_c


answer = False


def search(k):
    # Поиск следующей строки k в таблице A
    if not A:  # Если А не пуст
        for c in covered_cols:
            if not covered_cols[c]:  # тупик - решение
                answer = True
                print_solution()
                return
        return
    c = choose_col()
    # Выбор строк в которых есть тот самый столбец с минимом едениц
    rows = [node[0] for node in A if node[1] == c]
    if not rows:  # тупик
        return
    for r in rows:
        box = []  # Запоминаем удалённые строки
        # Добавляем строку r в наше решение
        B[k] = [node for node in A if node[0] == r]
        # Удаляем r из A.
        for node in B[k]:
            box.append(node)
            A.remove(node)
            updates[k] = updates.get(k, 0) + 1
        # Выбераем столбцы где j -> A[r,j]==1
        cols = [node[1] for node in B[k]]
        for j in cols:
            covered_cols[j] = True
            # Теперь строки i где A[i,j]==1.
            rows2 = [node[0] for node in A if node[1] == j]
            # Удаляем i из A памяти удалённых.
            tmp = [node for node in A if node[0] in rows2]
            for node in tmp:
                box.append(node)
                A.remove(node)
                updates[k] = updates.get(k, 0) + 1
        search(k + 1)
        # Восстанавливаем удалённые строки
        for node in box:
            A.append(node)
        del box
        del B[k]
        # Восстанвливаем столбцы
        for j in cols:
            covered_cols[j] = False
    return


search(0)
if not answer:
    print("FALSE")
time_passed()
