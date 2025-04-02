# Параллельное умножение матрицы на вектор
Программа содержит реализацию последовательного и параллельного умножения матрицы на вектор с использованием OpenMP.

## Описание проекта

-Генерация случайной квадратной матрицы и вектора -> generate(int r, int c)

-Последовательное умножение матрицы на вектор -> multiply(const matrix& a, const matrix& b)

-Параллельное умножение с использованием OpenMP -> multiply_parallel(const matrix& a, const matrix& b)

-Сравнение производительности последовательной и параллельной версий -> compare(const matrix& a, const matrix& b)
## Запуск программы
g++ -fopenmp -o main main.cpp

./main.exe
## Результат программы
| Вид умножения | Время выполнения |
|-------------|-----------------:|
| SERIAL      | 1245 ms          |
| PARALLEL    | **387 ms**       |
