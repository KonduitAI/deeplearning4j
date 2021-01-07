import json
import subprocess
import re
import os
import sys
from multiprocessing import Pool
cwd = os.getcwd()
try:
    os.mkdir("hip_errors")
except:
    pass
print("____"+cwd)
HIP_CMD = "/home/rgurbanov/HIPIFY/bin/./hipify-clang"
HIP_ARGS = ["--cuda-path=/usr/local/cuda",
            "--skip-excluded-preprocessor-conditional-blocks"]
pattern = re.compile(r"-o\s.*\.(cu|cpp)\.o")
pattern_cmp = re.compile(r"/usr/local/cuda/bin/nvcc")
pattern_arch = re.compile(r"arch=compute_\d+,code=sm_\d+")
replace_str = ["--cudart=static",
               "--expt-extended-lambda",
               "-forward-unknown-to-host-compiler",
               "-compress-all", "-fmax-errors=2",
               "-gencode", "-x cu", "-Xfatbin", "-Xcompiler=-fPIC "
               ]
exclude_dirs = [
    "/blasbuild/cuda/flatbuffers-build",
    "/blasbuild/cuda/tests_cpu/googletest-build/googlemock",
    "/blasbuild/cuda/tests_cpu/googletest-build/googletest"
]


def gen_cmd(cmd, fileX):
    cmd = cmd.replace("-c "+fileX, "")
    for rpl in replace_str:
        cmd = cmd.replace(rpl, "")
    cmd = cmd.replace("-isystem=", "-I")
    cmd = pattern_cmp.sub("", cmd)
    cmd = pattern.sub("", cmd)
    cmd = pattern_arch.sub("", cmd)
    args = [HIP_CMD]+HIP_ARGS+[fileX, "--"] + cmd.split()
    return args


def hipify_clang_call(args):
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
    return process.communicate()

def hipify(params):
    cmd, fileX = params
    args = gen_cmd(cmd, fileX)
    call_str = " ".join(args)
    print(call_str)
    out, err = hipify_clang_call(args)
    if len(err) > 0:
        with open("hip_errors/"+fileX.replace(cwd, "").replace("/", "_")+".txt", "w") as fw:
            fw.write(call_str+"\n\n")
            fw.write(err)

js= None
with open("blasbuild/cuda/compile_commands.json") as f:
    js = json.load(f)

process_list=[]
for each in js:
    cmd = each["command"]
    dirX = each["directory"].replace(cwd, "")
    fileX = each["file"]
    if dirX in exclude_dirs:
        print(f"excluded: {dirX}")
        continue
    if fileX.endswith(".cpp"):
        print(f"excluded: {fileX}")
        continue
    process_list.append((cmd, fileX))

with Pool(processes=16) as processors:
    processors.map(hipify, process_list)


