# Eco_Monitoring_Hack

![Eco_Monitoring_Hack Logo](link/to/your/logo.png)

**Репозиторий команды "Уральские Мандарины" для хакатона "Цифровой прорыв". Кейс: обработка видеофиксации транспорта.**

## Предсказание видео

Для предсказания видео выполните следующие шаги:

1. **Клонировать репозиторий:**
   ```bash
   git clone https://github.com/RomanLazovskiy/Eco_Monitoring_Hack
  ```
2. Скачайте файл модели 'https://drive.google.com/file/d/1TWfa1oYm2-VVV5QXD8j-Ny0xor2r4ljc/view?usp=sharing'
3. В файле process_video.py в переменную model_path укажите путь до файла модели.
4. В переменную markup_path укажите путь до папки с разметкой для видео.
5. Откройте ноутбук predict_video.ipynb
6. В переменную video_dir укажите путь до папки с видео
7. В переменную multiproc_video_count укажите количество парраллельных потоков с параллельной обработкой видео (1 поток = 1 видео)
8. В переменную result_file_name укажите нужное вам название выходное файла.
9. Запустите остальные ячейки ноутбука.

Выходной формат файла .csv будет состоять из колонок с такими именами:
file_name	quantity_car, average_speed_car,	quantity_van,	average_speed_van	quantity_bus,	average_speed_bus.

    
