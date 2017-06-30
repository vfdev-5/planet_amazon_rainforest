
import os
from datetime import datetime


from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
import keras.backend as K


from data_utils import GENERATED_DATA


# option= None,
# normalize_data = True,
# normalization = '',
# batch_size = 16,
# nb_epochs = 10,
# image_size = (224, 224),
# lrate_decay_f = None,
# samples_per_epoch = 2048,
# nb_val_samples = 1024,
# xy_provider_cache = None,
# class_weight = {},
# seed = None,
# save_prefix = "",
# verbose = 1


def write_info(filename, **kwargs):
    with open(filename, 'w') as f:
        for k in kwargs:
            f.write("{}: {}\n".format(k, kwargs[k]))


def classification_train(model,
                         train_id_type_list,
                         val_id_type_list,
                         **params):

    assert 'batch_size' in params, "Need batch_size"
    assert 'save_prefix' in params, "Need save_prefix"

    samples_per_epoch = len(train_id_type_list) if 'samples_per_epoch' not in params else params['samples_per_epoch']
    nb_val_samples = len(val_id_type_list) if 'nb_val_samples' not in params else params['nb_val_samples']
    lrate_decay_f = None if 'lrate_decay_f' not in params else params['lrate_decay_f']

    save_prefix = params['save_prefix']
    batch_size = params['batch_size']

    samples_per_epoch = (samples_per_epoch // batch_size + 1) * batch_size
    nb_val_samples = (nb_val_samples // batch_size + 1) * batch_size

    weights_path = os.path.join(GENERATED_DATA, "weights")
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    weights_filename = os.path.join(weights_path,
                                    save_prefix + "_{epoch:02d}_loss={loss:.4f}_val_loss={val_loss:.4f}")
    for mname in model.metrics:
        weights_filename += "_%s={%s:.4f}" % (mname, mname)
    weights_filename += ".h5"

    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)
    now = datetime.now()
    info_filename = os.path.join(weights_path,
                                 'training_%s_%s.info' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M"))))

    lr_base = model.optimizer.lr.eval(session=K.get_session()) if K.backend() == 'tensorflow' else model.optimizer.lr.eval()
    write_info(info_filename, params)

    csv_logger = CSVLogger('weights/training_%s_%s.log' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M"))))
    callbacks = [model_checkpoint, csv_logger, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))
    print("\n-- Fit model")
    try:
        if option == 'cervix/os':
            train_gen1, train_flow1 = get_train_gen_flow(train_id_type_list=train_id_type_list,
                                                         normalize_data=normalize_data,
                                                         normalization=normalization,
                                                         batch_size=batch_size,
                                                         seed=seed,
                                                         image_size=image_size,
                                                         option='cervix',
                                                         save_prefix=save_prefix,
                                                         xy_provider_cache=xy_provider_cache,
                                                         verbose=verbose)

            val_gen1, val_flow1 = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                                   normalization=normalization,
                                                   save_prefix=save_prefix,
                                                   normalize_data=normalize_data,
                                                   batch_size=batch_size,
                                                   seed=seed,
                                                   image_size=image_size,
                                                   option='cervix',
                                                   xy_provider_cache=xy_provider_cache)

            train_gen2, train_flow2 = get_train_gen_flow(train_id_type_list=train_id_type_list,
                                                         normalize_data=normalize_data,
                                                         normalization=normalization,
                                                         batch_size=batch_size,
                                                         seed=seed,
                                                         image_size=image_size, #tuple([int(s/2) for s in image_size]),
                                                         option='os',
                                                         save_prefix=save_prefix,
                                                         xy_provider_cache=xy_provider_cache,
                                                         verbose=verbose)

            val_gen2, val_flow2 = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                                   normalization=normalization,
                                                   save_prefix=save_prefix,
                                                   normalize_data=normalize_data,
                                                   batch_size=batch_size,
                                                   seed=seed,
                                                   image_size=image_size, #tuple([int(s/2) for s in image_size]),
                                                   option='os',
                                                   xy_provider_cache=xy_provider_cache)

            train_flow = map(lambda t: ([t[0][0], t[1][0]], t[0][1]), zip(train_flow1, train_flow2))
            val_flow = map(lambda t: ([t[0][0], t[1][0]], t[0][1]), zip(val_flow1, val_flow2))
        else:
            train_gen, train_flow = get_train_gen_flow(train_id_type_list=train_id_type_list,
                                                       normalize_data=normalize_data,
                                                       normalization=normalization,
                                                       batch_size=batch_size,
                                                       seed=seed,
                                                       image_size=image_size,
                                                       option=option,
                                                       save_prefix=save_prefix,
                                                       xy_provider_cache=xy_provider_cache,
                                                       verbose=verbose)

            val_gen, val_flow = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                                 normalization=normalization,
                                                 save_prefix=save_prefix,
                                                 normalize_data=normalize_data,
                                                 batch_size=batch_size,
                                                 seed=seed,
                                                 image_size=image_size,
                                                 option=option,
                                                 xy_provider_cache=xy_provider_cache)

        # New or old Keras API
        if int(keras_version[0]) == 2:
            print("- New Keras API found -")
            history = model.fit_generator(generator=train_flow,
                                          steps_per_epoch=(samples_per_epoch // batch_size),
                                          epochs=nb_epochs,
                                          validation_data=val_flow,
                                          validation_steps=(nb_val_samples // batch_size),
                                          callbacks=callbacks,
                                          class_weight=class_weight,
                                          verbose=verbose)
        else:
            history = model.fit_generator(generator=train_flow,
                                          samples_per_epoch=samples_per_epoch,
                                          nb_epoch=nb_epochs,
                                          validation_data=val_flow,
                                          nb_val_samples=nb_val_samples,
                                          callbacks=callbacks,
                                          class_weight=class_weight,
                                          verbose=verbose)
        # # save the last
        # val_loss = history.history['val_loss'][-1]
        # kwargs = {}
        # for mname in model.metrics:
        #     key = 'val_' + mname
        #     kwargs[key] = history.history[key][-1]
        # weights_filename = weights_filename.format(epoch=nb_epochs,
        #                                            val_loss=val_loss, **kwargs)
        # model.save_weights(weights_filename)
        return history

    except KeyboardInterrupt:
        pass

