#! /usr/bin/env python

# Generate PBS job files to be run on a computing cluster.
#
# NOTE: This script is highly cluster-specific, and will need to be
# modified for clusters different from the MSU GPU cluster.


import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils import utils


PBS_TEMPLATE = '''#
#PBS -l walltime={time:s}
#PBS -l mem={mem:d}gb,vmem={vmem:d}gb
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l feature='gpgpu:intel14'
#
#PBS -m a
#PBS -j oe
#PBS -o {logfile_path:s}

echo "========================================================================"
echo "== PBS JOB COMMENCING AT: `date -u '+%Y-%m-%d %H:%M:%S'` (UTC)"
echo "========================================================================"
echo ""

echo "PBS Header"
echo "----------"
echo "#PBS -l walltime={time:s}"
echo "#PBS -l mem={mem:d}gb,vmem={vmem:d}gb"
echo "#PBS -l nodes=1:ppn=1:gpus=1"
echo "#PBS -l feature='gpgpu:intel14'"
echo "#"
echo "#PBS -m a"
echo "#PBS -j oe"
echo "#PBS -o {logfile_path:s}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "set up environment"
echo "------------------"
if [ -n "$PBS_O_WORKDIR" ]
then
    echo "cd $PBS_O_WORKDIR"
    cd $PBS_O_WORKDIR
    echo ""
fi

echo "export PATH={pythonexec_path:s}:$PATH"
export PATH={pythonexec_path:s}:$PATH
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "ulimit -a"
echo "---------"
ulimit -a
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

N_GPUS_PRESENT=$(nvidia-smi --list-gpus | wc -l || echo "0")
if (( N_GPUS_PRESENT > 0 ))
then
    echo "module load CUDA/6.0"
    echo "--------------------"
    module load CUDA/6.0
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""

    echo "pre-command nvidia-smi --list-gpus (+ aggregate ecc error info)"
    echo "---------------------------------------------------------------"
    for ((i=0; i<N_GPUS_PRESENT; i++))
    do
        nvidia-smi --list-gpus | grep "GPU $i:"
        nvidia-smi -i $i -q | grep -A 2 -i "ecc mode"
        nvidia-smi -i $i -q | grep -i "ecc errors"
        nvidia-smi -i $i -q | grep -A 30 -i "ecc errors" | grep -A 14 -i "aggregate"
    done
    nvidia-smi --list-gpus
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
fi

echo "lscpu"
echo "-----"
lscpu
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "env | grep PBS"
echo "---------------"
env | grep PBS
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "cat \$PBS_NODEFILE"
echo "-----------------"
[ -f "$PBS_NODEFILE" ] && cat $PBS_NODEFILE || echo "<no PBS_NODEFILE>"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

if (( N_GPUS_PRESENT > 0 ))
then
    echo "cat \$PBS_GPUFILE"
    echo "-----------------"
    [ -f "$PBS_GPUFILE" ] && cat $PBS_GPUFILE || echo "<no PBS_GPUFILE>"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
fi

echo "echo the command to be run"
echo "--------------------------"
echo 'time {command:s}'
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "running command..."
echo "------------------"
time {command:s}
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

N_GPUS_PRESENT=$(nvidia-smi --list-gpus | wc -l || echo "0")
if (( N_GPUS_PRESENT > 0 ))
then
    echo "post-command nvidia-smi --list-gpus (+ aggregate ecc error info)"
    echo "----------------------------------------------------------------"
    for ((i=0; i<N_GPUS_PRESENT; i++))
    do
        nvidia-smi --list-gpus | grep "GPU $i:"
        nvidia-smi -i $i -q | grep -A 2 -i "ecc mode"
        nvidia-smi -i $i -q | grep -i "ecc errors"
        nvidia-smi -i $i -q | grep -A 30 -i "ecc errors" | grep -A 14 -i "aggregate"
    done
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
fi

echo "========================================================================"
echo "== PBS JOB COMPLETED AT: `date -u '+%Y-%m-%d %H:%M:%S'` (UTC)"
echo "========================================================================"
echo ""

if [ -n "$PBS_JOBID" ]
then
    echo "qstat -f $PBS_JOBID"
    echo "-------------------"
    qstat -f $PBS_JOBID
fi
'''

LLR_COMMAND_TEMPLATE = (
    '{analysis_script:s}'
    ' --template-settings="{template_settings:s}"'
    ' --minimizer-settings="{minimizer_settings:s}"'
    ' --ntrials={numtrials_per_job:d}'
    ' --outfile="{outfile_path:s}"'
    ' {flags:s}'
)

ASIMOV_COMMAND_TEMPLATE = (
    '{analysis_script:s}'
    ' --template-settings="{template_settings:s}"'
    ' --minimizer-settings="{minimizer_settings:s}"'
    ' --pseudo-data-settings="{pseudo_data_settings:s}"'
    ' --metric {metric:s}'
    ' --outfile="{outfile_path:s}"'
    ' {flags:s}'
)


if __name__ == "__main__":
    job_generation_timestamp = utils.timestamp(utc=True, tz=False, winsafe=True)

    parser = ArgumentParser(
        'Generate PBS job files for an analysis; submit with e.g. qsub_wrapper.sh.',
        #formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--analysis', choices=['llr', 'asimov'],
        help='Analysis to be performed.'
    )
    parser.add_argument(
        '--template-settings',
        required=True,
        help='Settings file to use for template making.'
    )
    parser.add_argument(
        '--minimizer-settings',
        required=True,
        help='minimizer settings file.'
    )
    parser.add_argument(
        '--single-octant', action='store_true',
        help='single octant in llh only.'
    )

    parser.add_argument(
        '--metric', choices=['chisquare', 'llh'],
        help='[Asimov only] Name of metric to use.'
    )
    parser.add_argument(
        '--pseudo-data-settings',
        metavar='JSONFILE', default=None,
        help='[Asimov only] Settings for pseudo data templates, if desired to'
             ' be different from template_settings.'
    )

    parser.add_argument(
        '--no-alt-fit', action='store_true',
        help='[LLR only] No alt hierarchy fit.'
    )
    parser.add_argument(
        '--numjobs', type=int, default=1,
        help='[LLR only] Number of job files to generate.'
    )
    parser.add_argument(
        '--numtrials-per-job', type=int, default=1,
        help='[LLR only] Number of LLR trials per job.'
    )

    parser.add_argument(
        '--time',
        type=str,
        default='00:20:00',
        help='Requested walltime per job, in format "hh:mm:ss".'
    )
    parser.add_argument(
        '--mem',
        type=int,
        default=4,
        help='Amount of physical memory to request, in GB.')
    parser.add_argument(
        '--vmem',
        type=int,
        default=4,
        help='Amount of virtual to request, in GB.')
    parser.add_argument(
        '--basedir',
        type=str,
        help='''Base directory. Follwing directory strcture is constructed:
Job files are placed in
    <basedir>/jobfiles
Results (json files) are placed in
    <basedir>/<analysis>__<ts basename>__<ms base>/results_rawfiles
Logfiles (*.err, *.out, and PBS *.log -- some or all of which may be
combined) are placed in
    <basedir>/<analysis>__<ts basename>__<ms base>/logfiles
Note that <ts basename>  and <ms basename> are the template and
minimizer settings files' basenames (i.e., without extensions),
respectively.'''
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=None,
        help='Set verbosity level.'
    )
    args = parser.parse_args()

    args.analysis = args.analysis.strip().lower()
    if args.analysis == 'llr':
        args.analysis_script = utils.expandPath(
            '$PISA/pisa/analysis/llr/LLROptimizerAnalysis.py'
        )
        command_template = LLR_COMMAND_TEMPLATE
        assert args.metric is None, \
                '"--metric" option not supported for LLR analysis.'
        assert args.pseudo_data_settings is None, \
                '"--pseudo-data-settings" option not supported for LLR analysis.'
    if args.analysis == 'asimov':
        args.analysis_script = utils.expandPath(
            '$PISA/pisa/analysis/asimov/AsimovOptimizerAnalysis.py'
        )
        command_template = ASIMOV_COMMAND_TEMPLATE
        assert args.no_alt_fit == False, \
                '"--no-alt-fit" flag not supported for Asimov analysis.'
        assert args.numjobs == 1, \
                'numjobs = %d invalid for Asimov analysis (only one outcome possible).' \
                %args.numjobs
        assert args.numtrials_per_job == 1, \
                'numtrials-per-job = %d invalid for Asimov analysis' \
                ' (only one outcome possible).' %args.numtrials_per_job

    if args.pseudo_data_settings is None:
        args.pseudo_data_settings = args.template_settings

    flags = ' --save-steps'
    if args.single_octant:
        flags += ' --single_octant'
    if args.no_alt_fit:
        flags += ' --no-alt-fit'
    args.flags = flags

    args.pythonexec_path = os.path.dirname(sys.executable)

    # Formulate file names from template and minimizer settings file names,
    # timestamp now, (and later, the file number in this sequence)
    ts_base, _ = os.path.splitext(os.path.basename(args.template_settings))
    ms_base, _ = os.path.splitext(os.path.basename(args.minimizer_settings))
    pd_base, _ = os.path.splitext(os.path.basename(args.pseudo_data_settings))

    if pd_base == ts_base:
        an_ts_ms_basename = '%s__%s__%s' %(args.analysis, ts_base, ms_base)
    else:
        an_ts_ms_basename = '%s__%s__%s__%s' %(args.analysis, ts_base,
                                               pd_base, ms_base)

    subdir = an_ts_ms_basename
    batch_basename = '%s__%s' %(an_ts_ms_basename, job_generation_timestamp)

    args.jobdir = utils.expandPath(
        os.path.join(args.basedir, 'jobfiles'), absolute=False
    )
    args.outdir = utils.expandPath(
        os.path.join(args.basedir, subdir, 'results_rawfiles'), absolute=False
    )
    args.logdir = utils.expandPath(
        os.path.join(args.basedir, subdir, 'logfiles'), absolute=False
    )

    # Create the required dirs (recursively) if they do not already exist
    utils.mkdir(args.jobdir)
    utils.mkdir(args.outdir)
    utils.mkdir(args.logdir)

    # Expand any variables in template/minimizer settings resource specs passed
    # in by user (do *not* make absoulte, since resource specs in PISA allow
    # for implicit referencing from the $PISA/pisa/resources dir)
    args.template_settings = utils.expandPath(args.template_settings,
                                              absolute=False)
    args.minimizer_settings = utils.expandPath(args.minimizer_settings,
                                               absolute=False)
    args.pseudo_data_settings = utils.expandPath(args.pseudo_data_settings,
                                                 absolute=False)

    for file_num in xrange(args.numjobs):
        job_basename = '%s__%06d' %(batch_basename, file_num)

        args.jobfile_path = os.path.join(args.jobdir, job_basename + '.pbs')
        args.logfile_path = os.path.join(args.logdir, job_basename + '.log')
        args.outfile_path = os.path.join(args.outdir, job_basename + '.json')

        args.command = command_template.format(**vars(args))

        pbs_text = PBS_TEMPLATE.format(**vars(args))

        print 'Writing %d bytes to "%s"' %(len(pbs_text), args.jobfile_path)
        with open(args.jobfile_path, 'w') as jobfile:
            jobfile.write(pbs_text)

    print '\n\n'
    print 'Command: %s' %args.command
    print '\n'
    print 'Finished creating %d PBS job files in directory %s' \
            %(args.numjobs, args.jobdir)
