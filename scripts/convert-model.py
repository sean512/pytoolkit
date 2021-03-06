#!/usr/bin/env python3
"""*.h5 を読んでONNXなどに変換するスクリプト。"""
import argparse
import os
import pathlib
import sys

if True:
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    import pytoolkit as tk


def _main():
    tk.utils.better_exceptions()
    tk.log.init(None)
    parser = argparse.ArgumentParser(description="*.h5 を読んでONNXなどに変換するスクリプト。")
    parser.add_argument(
        "mode", choices=("saved_model", "onnx", "tflite"), help="変換先の形式"
    )
    parser.add_argument("model_path", type=pathlib.Path, help="対象ファイルのパス(*.h5)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "none"
    with tk.dl.session():
        tk.K.set_learning_phase(0)

        tk.log.get(__name__).info(f"{args.model_path} Loading...")
        model = tk.models.load(args.model_path)

        if args.mode == "saved_model":
            save_path = args.model_path.with_suffix("")
        elif args.mode == "onnx":
            save_path = args.model_path.with_suffix(".onnx")
        elif args.mode == "tflite":
            save_path = args.model_path.with_suffix(".tflite")
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        tk.log.get(__name__).info(f"{save_path} Saving...")
        tk.models.save(model, save_path, mode=args.mode)

        tk.log.get(__name__).info("Finished!")


if __name__ == "__main__":
    _main()
