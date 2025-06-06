# Создание панорамы 

## Цель
Программа выполняет склейку двух изображений в панораму с использованием:
- Детектора ORB для поиска ключевых точек
- Метода brute-force matching для сопоставления точек
- RANSAC для вычисления гомографии
- Перспективного преобразования для объединения изображений

## Основные функции и алгоритмы

### Используемые функции OpenCV:
```
ORB::create()       # Создание детектора ключевых точек
detectAndCompute()  # Поиск ключевых точек и вычисление дескрипторов
BFMatcher()         # Сопоставление дескрипторов методом "грубой силы"
knnMatch()          # Поиск k наилучших соответствий
findHomography()    # Вычисление матрицы гомографии
warpPerspective()   # Применение перспективного преобразования
```

### Ключевые этапы обработки:
* Поиск ключевых точек (ORB)
* Фильтрация совпадений (Lowe's ratio test)
* Вычисление гомографии (RANSAC)
* Склейка изображений
* Обрезка черных областей

## Ввод и вывод
### Входные данные:
```
./part1.png       # Первое изображение (любой формат, поддерживаемый OpenCV)
./part2.png       # Второе изображение (должно перекрываться с первым)
```
### Выходные данные:
```
01_original1.jpg       # Исходное первое изображение
02_original2.jpg       # Исходное второе изображение
03_keypoints1.jpg      # Ключевые точки на первом изображении
04_keypoints2.jpg      # Ключевые точки на втором изображении
05_all_matches.jpg     # Визуализация всех совпадений
06_homography.jpg      # Визуализация преобразования
07_warped_image.jpg    # Второе изображение после преобразования
08_combined_panorama.jpg # Панорама до обрезки
09_final_panorama.jpg  # Финальный результат
```
