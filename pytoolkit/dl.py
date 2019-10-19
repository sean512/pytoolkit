"""DeepLearning(主にKeras)関連。"""
import functools
import os
import pathlib
import subprocess
import sys
import time
from operator import itemgetter

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from . import K



def wrap_session(config=None, gpu_options=None, use_horovod=False):
    """session()のデコレーター版。"""

    def decorator(func):
        @functools.wraps(func)
        def session_func(*args, **kwargs):
            with session(
                config=config, gpu_options=gpu_options, use_horovod=use_horovod
            ):
                return func(*args, **kwargs)

        return session_func

    return decorator


def session(config=None, gpu_options=None, use_horovod=False):
    """TensorFlowのセッションの初期化・後始末。

    使い方::

        with tk.dl.session():
            # kerasの処理


    Args:
        use_horovod: tk.hvd.init()と、visible_device_listの指定を行う。

    """

    class SessionScope:  # pylint: disable=R0903
        def __init__(self, config=None, gpu_options=None, use_horovod=False):
            self.config = config or {}
            self.gpu_options = gpu_options or {}
            self.use_horovod = use_horovod
            self.session = None

        def __enter__(self):
            if self.use_horovod:
                if tk.hvd.initialized():
                    tk.hvd.barrier()  # 初期化済みなら初期化はしない。念のためタイミングだけ合わせる。
                else:
                    tk.hvd.init()
                if tk.hvd.initialized() and get_gpu_count() > 0:
                    self.gpu_options["visible_device_list"] = str(
                        tk.hvd.get().local_rank()
                    )
            if K.backend() == "tensorflow":
                self.config["allow_soft_placement"] = True
                self.gpu_options["allow_growth"] = True
                if (
                    "OMP_NUM_THREADS" in os.environ
                    and "intra_op_parallelism_threads" not in self.config
                ):
                    self.config["intra_op_parallelism_threads"] = int(
                        os.environ["OMP_NUM_THREADS"]
                    )
                config = tf.compat.v1.ConfigProto(**self.config)
                for k, v in self.gpu_options.items():
                    setattr(config.gpu_options, k, v)
                self.session = tf.compat.v1.Session(config=config)
                K.set_session(self.session)
            return self

        def __exit__(self, *exc_info):
            if K.backend() == "tensorflow":
                self.session = None
                K.clear_session()

    return SessionScope(config=config, gpu_options=gpu_options, use_horovod=use_horovod)


def get_gpu_count():
    """GPU数の取得。"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if gpus in ("-1", "none"):
            return 0
        return len(np.unique(gpus.split(",")))
    try:
        result_text = nvidia_smi("--list-gpus").strip()
        if "No devices found" in result_text:
            return 0
        return len([l for l in result_text.split("\n") if len(l) > 0])
    except FileNotFoundError:
        return 0


def nvidia_smi(*args):
    """nvidia-smiコマンドを実行する。"""
    path = (
        pathlib.Path(os.environ.get("ProgramFiles", ""))
        / "NVIDIA Corporation"
        / "NVSMI"
        / "nvidia-smi.exe"
    )
    if not path.is_file():
        path = "nvidia-smi"
    command = [str(path)] + list(args)
    return subprocess.check_output(
        command, stderr=subprocess.STDOUT, universal_newlines=True
    )





def dontuse_GPU_list(use_gpus=2, select_mode='top'):
    """
    do not use
    
    https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
    ↑に切り替えた方がええかな?
    
    共有マシンなどでGPUが複数ある環境で、使われているGPUを判断する
    
    gpus = gpu_list.GPU_list(use_gpus=2)
    num_gpus = len(gpus)
    if sys.platform.startswith("win32"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus]) if num_gpus else '-1'
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    elif sys.platform.startswith("linux"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    tf_config = tf.ConfigProto(device_count={'GPU': len(gpus)})
    :param use_gpus: 使いたいgpuの数
    :param select_mode: 未実装
    :return:


    gpuが複数あったら
    index,gpu利用率,メモリー利用率を出す

    選択モードに沿って使う数だけ抜き出す
    gpu利用率,メモリー利用率が40%以下だったらそのまま使用
    ~~~1秒間隔で2回再調査~~~
    """
    run_cmd = "nvidia-smi -L"
    try:
        smi_runout_raw = subprocess.run(run_cmd.split(), shell=False, stdout=subprocess.PIPE, universal_newlines=True)
    except:
        return []
    # print(smi_run_out.stdout)
    smi_out = [gpu.split(' ', maxsplit=1) for gpu in smi_runout_raw.stdout.splitlines()]
    smi_out = [[gpu[0]] + gpu[1].split(': ', maxsplit=1) for gpu in smi_out]  # memo なんかいやだ
    print(smi_runout_raw.stdout, end='')
    # pprint.pprint(smi_out,depth=2,width=140)#print(smi_out)
    if 1 <= len(smi_out) <= use_gpus:  # len(smi_out)==1 or 1<len(smi_out)<use_gpus:
        gpu_stats = get_gpu_info()
        return dontuse_select_gpu(use_gpus, gpu_stats, select_mode)
        # pass# GPUが1個以上use_gpus以下の時の処理
    elif len(smi_out) > use_gpus:
        gpu_stats = get_gpu_info()
        return dontuse_select_gpu(use_gpus, gpu_stats, select_mode)
        # pass#GPUがuse_gpus個以上あったとき
    else:  # 万が一,GPUが見つからなかったら
        raise SystemError("GPU is Not Found")
    # return smi_runout_raw.stdout,smi_out


def dontuse_select_gpu(use_gpus, gpu_info, select_mode='top'):
    """
    バグがあるっぽいのでdo not use
    2018年に慌てて作ったコードなので汚い
    何をしているかうろ覚え..."""
    temp_limt = 60  # あちあちと判定する閾値
    memoryfree_limt = 4096 if sys.platform.startswith("win32") else 8192  # 10240
    utilization_gpu_limit = 40
    utilization_memory_limit = 40
    rescan_num = 2  # 再スキャンする回数
    rescan_time = 2  # 再スキャン間隔(秒)
    if select_mode == 'top':
        print("top mode")
        favorite_gpus = gpu_info[:use_gpus]
        favorite_gpu_useflag = []
        for gpu in favorite_gpus:
            use_flag = 1
            use_flag = use_flag if gpu['memory.free'] > memoryfree_limt else 0
            use_flag = use_flag if gpu['temperature.gpu'] < temp_limt else 0
            use_flag = use_flag if gpu['utilization.gpu'] < utilization_gpu_limit else 0
            use_flag = use_flag if gpu['utilization.memory'] < utilization_memory_limit else 0
            if use_flag == 1:  # memo 再スキャン結果okなら
                favorite_gpu_useflag.append(1)
            else:  # memo 再スキャン結果ダメなら
                favorite_gpu_useflag.append(0)
            # favorite_gpu_useflag.append(use_flag)
        bad_gpus = [i for i, x in enumerate(favorite_gpu_useflag) if x == 0]  # memo 規定値以上だったgpuがあるか
        good_gpus = [i for i, x in enumerate(favorite_gpu_useflag) if x == 1]
        if bad_gpus:
            print("select GPU bad 1")
            favorite_gpu_useflag2 = []  # itertools.product
            favorite_gpus2 = get_gpu_info(','.join(map(str, bad_gpus)))  # 不安str
            for i, gpu in enumerate(favorite_gpus2):
                use_flag = 1
                use_flag = use_flag if gpu['memory.free'] > memoryfree_limt else 0
                use_flag = use_flag if gpu['temperature.gpu'] < temp_limt else 0
                use_flag = use_flag if gpu['utilization.gpu'] < utilization_gpu_limit else 0
                use_flag = use_flag if gpu['utilization.memory'] < utilization_memory_limit else 0
                if use_flag == 1:  # memo 再スキャン結果okなら
                    favorite_gpu_useflag2.append(1)
                    good_gpus.append(bad_gpus[i])
                else:  # memo 再スキャン結果ダメなら
                    favorite_gpu_useflag2.append(0)
            bad_gpus2 = [i for i, x in enumerate(favorite_gpu_useflag2) if x == 0]  # memo 規定値以上だったgpuがあるか
            if bad_gpus2:
                print("select GPU bad 2")
                bad_gpus3 = bad_gpus2
                for i in range(rescan_num - 1):
                    time.sleep(rescan_time)
                    notuse_gpulist = [bad_gpus[i_2] for i_2 in bad_gpus3]
                    favorite_gpus3 = get_gpu_info(','.join(map(str, notuse_gpulist)))  # 不安str
                    for i_3, gpu in enumerate(favorite_gpus3):
                        use_flag = 1
                        use_flag = use_flag if gpu['memory.free'] > memoryfree_limt else 0
                        use_flag = use_flag if gpu['temperature.gpu'] < temp_limt else 0
                        use_flag = use_flag if gpu['utilization.gpu'] < utilization_gpu_limit else 0
                        use_flag = use_flag if gpu['utilization.memory'] < utilization_memory_limit else 0
                        if use_flag == 1:  # memo 再スキャン結果okなら
                            bad_gpus3.pop(i_3)
                            good_gpus.append(notuse_gpulist[i_3])
                        # else:  # memo 再スキャン結果ダメなら
                        #    #favorite_gpu_useflag2.append(0)
                if bad_gpus3:  # memo リスキャンしてもダメだったら
                    print("select GPU bad 3")
                    nocheck_gpulist = [i for i in range(len(gpu_info))]
                    del nocheck_gpulist[:use_gpus]
                    if not nocheck_gpulist:
                        pass  # memo リスキャンしてもダメだったら
                    else:
                        # use_and_ng_gpulist=[i for i in range(len(gpu_info))]
                        # nocheck_gpulist=[i for i in range(len(gpu_info))]
                        notuse_gpulist = [bad_gpus[i] for i in bad_gpus3]
                        notuse_gpulist2 = [bad_gpus[i] for i in bad_gpus3]
                        favorite_gpus4 = get_gpu_info(','.join(map(str, nocheck_gpulist)))  # 不安str
                        # print(favorite_gpus4)
                        print(nocheck_gpulist)
                        plus_num = -1
                        for i, noset_gpu in enumerate(notuse_gpulist):
                            if not nocheck_gpulist:
                                break
                            print(nocheck_gpulist)
                            for i_2, gpu in enumerate(favorite_gpus4):
                                use_flag = 1
                                use_flag = use_flag if gpu['memory.free'] > memoryfree_limt else 0
                                use_flag = use_flag if gpu['temperature.gpu'] < temp_limt else 0
                                use_flag = use_flag if gpu['utilization.gpu'] < utilization_gpu_limit else 0
                                use_flag = use_flag if gpu['utilization.memory'] < utilization_memory_limit else 0
                                if use_flag == 1:  # memo 再スキャン結果okなら
                                    if plus_num >= i_2:  # 登録済みGPUを割り当てないように
                                        continue
                                    else:
                                        notuse_gpulist2.pop(i)
                                        nocheck_gpulist.pop(i_2)
                                        good_gpus.append(use_gpus + i_2)  # notuse_gpulist[i_2])
                                        # favorite_gpus4.pop(i_2)
                                        plus_num = plus_num + 1
                                        break
        
        # print(good_gpus)
        return good_gpus  # memo これでいいのだろうか?
    
    # =get_gpu_info()


def pick_gpus():
    """
    メモリー利用率とCPU利用率が低い順に出している
    温度も考慮したい

    :return:
    """
    out_list = get_gpu_info()
    # stats = {k: [v[k] for v in out_list] for k in DEF_REQ_DATA}
    stats = out_list
    import random
    ids = map(lambda gpu: int(gpu['index']), stats)
    memory_useratios = map(lambda gpu: float(gpu['memory.used']) / float(gpu['memory.total']), stats)
    gpu_useratios = map(lambda gpu: int(gpu['utilization.gpu']), stats)
    pairs = list(zip(ids, memory_useratios, gpu_useratios))
    random.shuffle(pairs)
    # bestGPU = min(pairs, key=lambda x: x[1])#[0]
    # bestGPUs = sorted(pairs, key=lambda x: x[1])  # [0]
    bestGPUs = sorted(pairs, key=itemgetter(1, 2))
    print(bestGPUs)
    bestGPUs = [i[0] for i in bestGPUs]
    
    return bestGPUs

def strtoint(val, lensize=6):
    if len(val) <= lensize:
        return int(val)
    else:
        return val


DEF_REQ_DATA = (
    'index', 'temperature.gpu', 'fan.speed', 'utilization.gpu', 'utilization.memory', 'memory.total', 'memory.free', 'memory.used', 'name',
    'uuid', 'timestamp')


def get_gpu_info(id=None, no_units=True,
                 req_data=DEF_REQ_DATA):  # (nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True,no_header=True):
    base_cmd = "nvidia-smi" if not id else "nvidia-smi" + " --id=" + id
    # req_data=keys
    nu_opt = '' if not no_units else ',nounits'
    cmd = base_cmd + " --query-gpu=" + ','.join(req_data) + " --format=csv,noheader" + nu_opt
    print(cmd)
    cmd_out = subprocess.run(cmd.split(), shell=False, stdout=subprocess.PIPE, universal_newlines=True)
    lines = cmd_out.stdout.splitlines()
    lines = [line.strip() for line in lines if line.strip() != '']
    out_list = [{k: strtoint(v) for k, v in zip(req_data, line.split(', '))} for line in lines]
    # lines = [ line.strip() for line in lines if line.strip() != '' ]
    #
    # return [ { k: v for k, v in zip(req_data, line.split(', ')) } for line in lines ]
    return out_list
