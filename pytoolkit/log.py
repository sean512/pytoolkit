"""ログ関連"""
import functools
import logging
import logging.handlers
import time


def get(name='__main__'):
    """"ロガーを取得して返す。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


def stream_handler(stream=None, level=logging.INFO, fmt='[%(levelname)-5s] %(message)s'):
    """StreamHandlerを作成して返す。"""
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    return handler


def file_handler(output_path, append=False, rotate=False,
                 max_bytes=1048576, backup_count=10, encoding='utf-8',
                 level=logging.DEBUG,
                 fmt='%(asctime)s [%(levelname)-5s] [%(filename)s:%(lineno)d] %(message)s'):
    """RotatingFileHandler / FileHandlerを作成して返す。"""
    if rotate:
        handler = logging.handlers.RotatingFileHandler(
            str(output_path), 'a', max_bytes, backup_count, encoding=encoding)
    else:
        handler = logging.FileHandler(str(output_path), 'a' if append else 'w', encoding=encoding)
    handler.setLevel(level)
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    return handler


def close(logger):
    """loggerが持っているhandlerを全部closeしてremoveする。"""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def trace(name='__main__'):
    """関数の開始・終了をログる。

    # 引数
    - name: ロガーの名前。

    """
    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            logger = get(name)
            logger.debug('%s 開始', func.__name__)
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_time = time.time() - start_time
                logger.debug('%s 終了 (time=%.3f)', func.__name__, elapsed_time)

        return _decorated_func
    return _decorator
