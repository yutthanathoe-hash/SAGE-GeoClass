# -*- coding: utf-8 -*-
"""Standalone TensorFlow worker. NEVER import GDAL, QGIS, or rasterio here.
CLI: python LandUse_TF_Worker_V2_4_4_6.py train|predict --request FILE --response FILE
Each invocation is a fresh process: h5py loads before tensorflow and any GIS DLLs.
"""
import argparse
import json
import os
import sys
import traceback

def run(command, request, response):
    import h5py  # CRITICAL: first binary scientific import
    import tensorflow as tf
    import numpy as np
    with np.load(request, allow_pickle=False) as data:
        if command == 'train':
            xtr = data['xtr'].astype('float32', copy=False)
            xva = data['xva'].astype('float32', copy=False)
            ytr = data['ytr'].astype('int32')
            yva = data['yva'].astype('int32')
            kind = str(data['kind'].item())
            epochs = int(data['epochs'].item())
            batch = int(data['batch'].item())
            seed = int(data['seed'].item())
            path = str(data['model_path'].item())
            tf.keras.utils.set_random_seed(seed)
            n_classes = int(data['n_classes'].item())
            if xtr.shape[0] == 0 or xva.shape[0] == 0:
                raise ValueError('Empty training or validation set')
            if kind == 'SpatialCNN':
                if xtr.ndim != 4 or xva.shape[1:] != xtr.shape[1:]:
                    raise ValueError('SpatialCNN expects (N,patch,patch,bands)')
                inp = tf.keras.Input(shape=xtr.shape[1:])
                x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inp)
                x = tf.keras.layers.MaxPooling2D(2)(x)
                x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            elif kind == 'TemporalCNN':
                if xtr.ndim != 3 or xtr.shape[1] < 2 or xva.shape[1:] != xtr.shape[1:]:
                    raise ValueError('TemporalCNN expects (N,time>=2,features)')
                inp = tf.keras.Input(shape=xtr.shape[1:])
                x = tf.keras.layers.Conv1D(64,3,padding='same',activation='relu')(inp)
                x = tf.keras.layers.Conv1D(64,3,padding='same',activation='relu')(x)
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
            elif kind == 'DNN':
                if xtr.ndim != 2 or xva.shape[1:] != xtr.shape[1:]:
                    raise ValueError('DNN expects (N,features)')
                inp = tf.keras.Input(shape=xtr.shape[1:])
                x = tf.keras.layers.Dense(128,activation='relu')(inp)
            else:
                raise ValueError('Unknown model type: ' + kind)
            x = tf.keras.layers.Dense(64,activation='relu')(x)
            x = tf.keras.layers.Dropout(.25)(x)
            out = tf.keras.layers.Dense(n_classes,activation='softmax')(x)
            model = tf.keras.Model(inp,out)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            model.fit(xtr,ytr,validation_data=(xva,yva),epochs=epochs,batch_size=batch,verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)])
            os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
            model.save(path)
            probabilities = model.predict(xva,batch_size=batch,verbose=0)
            np.savez_compressed(response, probabilities=probabilities)
        elif command == 'predict':
            x = data['x'].astype('float32',copy=False)
            path = str(data['model_path'].item())
            batch = int(data['batch'].item())
            model = tf.keras.models.load_model(path,compile=False)
            probabilities = model.predict(x,batch_size=batch,verbose=0)
            np.savez_compressed(response, probabilities=probabilities)
        else:
            raise ValueError(command)


def serve(model_path, batch):
    """Persistent JSONL protocol on stdout; diagnostics only to stderr."""
    import h5py  # must precede tensorflow
    import tensorflow as tf
    import numpy as np
    model = tf.keras.models.load_model(model_path, compile=False)
    print(json.dumps({"ok":True,"status":"ready"}), flush=True)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("command") == "shutdown":
                print(json.dumps({"ok":True,"status":"bye"}), flush=True)
                break
            if message.get("command") != "predict":
                raise ValueError("Unknown command")
            x = np.load(message["request"], allow_pickle=False)
            y = model.predict(x, batch_size=batch, verbose=0)
            batch_id = int(message["batch_id"])
            response_path = message["response"]
            staged_path = response_path + ".staging"
            with open(staged_path, "wb") as out:
                np.save(out, np.asarray(y, dtype=np.float32), allow_pickle=False)
                out.flush()
                os.fsync(out.fileno())
            if os.path.exists(response_path):
                raise FileExistsError("IPC response already exists: " + response_path)
            os.rename(staged_path, response_path)
            print(json.dumps({"ok":True,"rows":int(len(y)),"batch_id":batch_id}), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"ok":False,"error":str(exc)}), flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','predict','serve'])
    parser.add_argument('--request')
    parser.add_argument('--response')
    parser.add_argument('--model')
    parser.add_argument('--batch',type=int,default=128)
    args=parser.parse_args()
    try:
        if args.command == "serve":
            if not args.model: parser.error("--model is required for serve")
            serve(args.model,args.batch)
        else:
            if not args.request or not args.response: parser.error("--request and --response required")
            run(args.command,args.request,args.response)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
if __name__=='__main__':
    main()
