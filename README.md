# Dancing-Links-Polyomino-cover-dlx

Exact cover problem with Polyiomino. Only L-shape, and rectangle shape.
Using Dancing Links algorithm by Donald Knuth. https://arxiv.org/abs/cs/0011047 
-----------------------------------------------------------------------------------------------------------------------------
Решение покрытия поля полиомино при помощи Алгоритма Dancing Links Дональда кнута. Только L-фигуры и прямоугольники.
(Алгоритм находит не решение исключительно полного покрытия, будут и решения, где на доске остаются не покрытые клетки)
(M1, M2) – размер прямоугольника-стола T, тапл-пара положительных
целых чисел; 
((S1, S2), C)  - лист из тапл-пар, содержащий информацию об опорных прямоуголных полиомино. C - количество таких фигур на доске 
((L1, L2), C) – лист из тапл-пар, содержащий информацию об опорных L-полиомино. Первый элемент такой пары (L1) длина левой коёмки, второй (L2) - "нижней коемки". C - количество таких фигур на доске 

Сначла програма переводит данные в матрицу смежности.
После этого переводит матрицу смежности в лист смежности(ноды), а затем выполняется поиск при помощи алгоритма Кнута. 

С алгоритмом можно ознакомиться здесь - https://arxiv.org/abs/cs/0011047 

Хабр - https://habr.com/ru/post/194410/ 


![Polyomino puzzle](https://github.com/Audorion/Dancing-Links-Polyomino-cover-dlx/blob/master/example-8x8.svg)
