"""CLI関連。"""
from __future__ import annotations

import argparse
import logging
import pathlib
import typing

import pytoolkit as tk


class App:
    """MLコンペとか用簡易フレームワーク。

    ログの初期化とかのボイラープレートコードを出来るだけ排除するためのもの。

    Args:
        output_dir: ログ出力先ディレクトリ

    Fields:
        output_dir: ログ出力先ディレクトリ
        current_command: 現在実行中のコマンド名

    """

    def __init__(self, output_dir: typing.Union[tk.typing.PathLike, None]):
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.inits: typing.List[typing.Callable[[], None]] = [
            tk.utils.better_exceptions,
            tk.math.set_ndarray_format,
        ]
        self.terms: typing.List[typing.Callable[[], None]] = []
        self.commands: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
        self.current_command: typing.Optional[str] = None
        
        self.log_option_is=False

    def init(self):
        """前処理の追加用デコレーター"""

        def _decorator(func):
            self.inits.append(func)
            return func

        return _decorator

    def term(self):
        """後処理の追加用デコレーター"""

        def _decorator(func):
            self.terms.append(func)
            return func

        return _decorator
    
    def log_option(self,output_path="",append=False,rotate=False,max_bytes=1048576,backup_count=10,stream_level=logging.INFO,
                   stream_fmt="[%(levelname)-5s] %(message)s",file_level=logging.DEBUG,
                   file_fmt="%(asctime)s [%(levelname)-5s] %(message)s <%(name)s> %(filename)s:%(lineno)d",
                   matplotlib_level=logging.WARNING,pil_level=logging.INFO):
        """
        run実行時のログの設定、output_pathが空文字列だとself.output_dir / f"{command['func'].__name__}.log"のままにする
        全体的に美しくないのはゆるして
        """
        self.log_option_is = True
        self.log_output_path = output_path
        self.log_append = append
        self.log_rotate = rotate
        self.log_max_bytes = max_bytes
        self.log_backup_count = backup_count
        self.log_stream_level = stream_level
        self.log_stream_fmt = stream_fmt
        self.log_file_level = file_level
        self.log_file_fmt = file_fmt
        self.log_matplotlib_level = matplotlib_level
        self.log_pil_level = pil_level
    
    def command(self, logfile: bool = True, then: str = None):
        """コマンドの追加用デコレーター。

        Args:
            logfile: ログファイルを出力するのか否か。
            then: 当該コマンドが終わった後に続けて実行するコマンドの名前。

        """
        assert not logfile or self.output_dir is not None

        def _decorator(func):
            if func.__name__ in self.commands:
                raise ValueError(f"Duplicated command: {func.__name__}")
            self.commands[func.__name__] = {
                "func": func,
                "logfile": logfile,
                "then": then,
            }
            return func

        return _decorator

    def run(self, argv: list = None, default: str = None):
        """実行。

        Args:
            args: 引数。(既定値はsys.argv)
            default: 未指定時に実行するコマンド名 (既定値は先頭のコマンド)

        """
        commands = self.commands.copy()
        if "ipy" not in commands:
            commands["ipy"] = {"func": self._ipy, "logfile": False, "then": None}
        command_names = list(commands)
        default = default or command_names[0]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "command", choices=command_names, nargs="?", default=default
        )
        args = parser.parse_args(argv)

        self.current_command = args.command
        while True:
            assert self.current_command is not None
            command = commands[self.current_command]
            
            # 美しくないけどゆるして
            if self.log_option_is:
                final_outputname=self.output_dir / f"{command['func'].__name__}.log" if command["logfile"] and self.output_dir is not None \
                    else None if len(self.log_output_path) == 0 else self.log_output_path
                tk.log.init(final_outputname,append=self.log_append,rotate=self.log_rotate,max_bytes=self.log_max_bytes,
                            backup_count=self.log_backup_count,stream_fmt=self.log_stream_fmt,file_level=self.log_file_level,
                            file_fmt=self.log_file_fmt,matplotlib_level=self.log_matplotlib_level,pil_level=self.log_pil_level)
            else:
                # ログ初期化
                tk.log.init(
                    self.output_dir / f"{command['func'].__name__}.log"
                    if command["logfile"] and self.output_dir is not None
                    else None
                )
            # 前処理
            for f in self.inits:
                with tk.log.trace_scope(f.__qualname__):
                    f()
            try:
                with tk.log.trace_scope(command["func"].__qualname__):
                    command["func"]()
            except BaseException as e:
                # KeyboardInterrupt以外で、かつ
                # ログファイルを出力する(ような重要な)コマンドの場合のみ通知を送信
                if e is not KeyboardInterrupt and command["logfile"]:
                    tk.notifications.post(tk.utils.format_exc())
                raise
            finally:
                # 後処理
                for f in self.terms:
                    with tk.log.trace_scope(f.__qualname__):
                        f()

            # 次のコマンド
            self.current_command = command["then"]
            if self.current_command is None:
                break

    def _ipy(self):
        """自動追加されるコマンド。ipython。"""
        import sys
        import IPython

        m = sys.modules["__main__"]
        user_ns = {k: getattr(m, k) for k in dir(m)}
        IPython.start_ipython(argv=[], user_ns=user_ns)
