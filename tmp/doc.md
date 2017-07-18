# Как работает мой пайплайн :

Основные идеи :
- Автоматическое логирование все экспериментов
- Модулярность

## Структура

 - common : общие функции, модели и данные для скриптов, тетрадок
 - notebooks : тетрадки с эскпериментами
 - scripts : скрпиты для запуска тренировок, валидаций и тестирования

## Данные

По умолчанию находятся в
```
input/train/jpg/*.tif
input/train/tif/*.tif
input/test/jpg/*.jpg
input/test/tif/*.tif
input/sample_submission_v2.csv
```

Корневую папку можно настроить через переменную среды `INPUT_PATH`

## Подача данных

Пайплайн написан для Keras и подразумевает использовать модифицированный ImageDataGenerator.

ImageDataGenerator использует спецализированную подачу данных `xy_provider`

Базовый пример использования :
```
def xy_provider(image_ids, infinite=True):
    while True:
        np.random.shuffle(image_ids)
        for image_id in image_ids:
            image = load_image(image_id)
            target = load_target(image_id)

            # Some custom preprocesssing: resize
            # ...
            yield image, target
        if not infinite:
            return

train_gen = ImageDataGenerator(pipeline=('random_transform', 'standardize'),
                         featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.15, height_shift_range=0.15,
                         shear_range=3.14/6.0,
                         zoom_range=0.25,
                         channel_shift_range=0.1,
                         horizontal_flip=True,
                         vertical_flip=True)

train_gen.fit(xy_provider(train_image_ids, infinite=False),
        len(train_image_ids),
        augment=True,
        save_to_dir=GENERATED_DATA,
        batch_size=4,
        verbose=1)

val_gen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True) # Just an infinite image/mask generator

val_gen.mean = train_gen.mean
val_gen.std = train_gen.std
val_gen.principal_components = train_gen.principal_components

history = model.fit_generator(
    train_gen.flow(xy_provider(train_image_ids), # Infinite generator is used
                   len(train_id_type_list),
                   batch_size=batch_size),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epochs,
    validation_data=val_gen.flow(xy_provider(val_image_ids), # Infinite generator is used
                   len(val_image_ids),
                   batch_size=batch_size),
    nb_val_samples=nb_val_samples)
```

**Модифицированный ImageDataGenerator принимает на вход лист операций для аугментации данных**
```
ImageDataGenerator(pipeline=('random_transform', 'standardize'), **kwargs)
```

## Тренировка

Чтобы

### training_utils.py

- classification_train


