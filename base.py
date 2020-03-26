# routines across all benchmarks, e.g. setting the tsinfer version
import os
import sys
import subprocess
import importlib
import inspect
import logging

def version_location():
    """
    Where to save different versions of the same library, which will be loaded as
    appropriate
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '_versions')

def check_version(repo, commit, commitkey_len=None):
    """
    E.g. check_version('http://github.com/tskit-dev/tsinfer', 'efbafff')
    Check that this module is installed, and return the installation dir and short commit hash.
    
    If `commit`=='' then we download and save under libname_
    and don't check the commit hash
    """
    
    if commitkey_len is not None:
        commit = commit[:commitkey_len]
    libname = repo.split("/")[-1]
    dirname = libname + "_" + (commit or '') 
    fulldir = os.path.join(version_location(), dirname)
    if os.path.isdir(fulldir):
        # already installed - check the right commit
        logging.debug("Already installed", commit, ": checking")
        if commit == "master":
            subprocess.call(["git", "fetch", "origin"], cwd=fulldir)
            subprocess.call(["git", "reset", "--hard", "origin/master"], cwd=fulldir)
        retcommit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=fulldir)
        retcommit = str(retcommit, "utf-8").strip()  # In case it is a byte string
        if commit in ('master', ''):
            return fulldir, retcommit
        else:
            assert commit.startswith(retcommit)
    else:
        # install the repo and check out the right commit
        retcode = subprocess.call(["git", "clone", "--recursive", repo,  fulldir])
        assert os.path.isdir(fulldir)
        assert retcode == 0
        if commit:
            retcode = subprocess.call(["git", "checkout", commit], cwd=fulldir)
            assert retcode == 0
        retcode = subprocess.call(["make"], cwd=fulldir)
        assert retcode == 0
        retcommit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=fulldir)
        retcommit = str(retcommit, "utf-8").strip()  # In case it is a byte string
    return fulldir, retcommit

def check_tsinfer_version(commit):
    return check_version('http://github.com/tskit-dev/tsinfer', commit, commitkey_len=7)
    
def import_tsinfer(commit):
    """
    Import a specific commit version of tsinfer, and return the name of the library
    loaded and the short commit hash. If '', do not check the commit hash.
    """
    oldpath = sys.path
    fulldir, commithash = check_tsinfer_version(commit)
    sys.path = [fulldir] + oldpath
    to_del = []
    for name, module in sys.modules.items():
        try:
            if module.__file__.startswith(version_location()):
                to_del.append(name)
        except AttributeError:
            pass  # builtins do not have a file attribute
    for name in to_del:
        del sys.modules[name]
    modulename = importlib.import_module('tsinfer')
    sys.path = oldpath
    return modulename, commithash


def ts_kc(ts1, ts2):
    assert ts1.sequence_length == ts2.sequence_length
    tree1_iter = ts1.trees()
    curr_pos = 0
    kc = 0
    tree1 = next(tree1_iter)
    for tree2 in ts2.trees():
        while tree1.interval[1] < tree2.interval[1]:
            span = tree1.interval[1] - curr_pos
            curr_pos = tree1.interval[1]
            kc += tree1.kc_distance(tree2) * span
            tree1 = next(tree1_iter)
        span = tree2.interval[1] - curr_pos
        curr_pos = tree2.interval[1]
        kc += tree1.kc_distance(tree2) * span
    assert tree1.interval[1] == curr_pos
    return kc / ts1.sequence_length


def time_cmd(cmd, stdout=sys.stdout):
    """
    Runs the specified command line (a list suitable for subprocess.call)
    and writes the stdout to the specified file object.
    """
    if sys.platform == 'darwin':
        #on OS X, install gtime using `brew install gnu-time`
        time_cmd = "/usr/local/bin/gtime"
    else:
        time_cmd = "/usr/bin/time"
    full_cmd = [time_cmd, "-f%M %S %U"] + cmd
    with tempfile.TemporaryFile() as stderr:
        exit_status = subprocess.call(full_cmd, stderr=stderr, stdout=stdout)
        stderr.seek(0)
        if exit_status != 0:
            raise ValueError(
                "Error running '{}': status={}:stderr{}".format(
                    " ".join(cmd), exit_status, stderr.read()))

        split = stderr.readlines()[-1].split()
        # From the time man page:
        # M: Maximum resident set size of the process during its lifetime,
        #    in Kilobytes.
        # S: Total number of CPU-seconds used by the system on behalf of
        #    the process (in kernel mode), in seconds.
        # U: Total number of CPU-seconds that the process used directly
        #    (in user mode), in seconds.
        max_memory = int(split[0]) * 1024
        system_time = float(split[1])
        user_time = float(split[2])
    return user_time + system_time, max_memory
