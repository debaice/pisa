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
echo "--------------"
env | grep PBS
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "cat \$PBS_NODEFILE"
echo "------------------"
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

echo "echo the command(s) to be run"
echo "--------------------00-------"
echo '{command_a:s} &'
echo '{command_b:s} &'
echo 'wait'
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""

echo "running command..."
echo "------------------"
{command_a:s} &
{command_b:s} &
wait
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


command_template = (
    '{analysis_script:s}'
    ' --template-settings="{template_settings:s}"'
    ' --minimizer-settings="{minimizer_settings:s}"'
    ' --ntrials={numtrials_per_job:d}'
    ' --outfile="{outfile_path:s}"'
    ' {flags:s}'
)


if __name__ == "__main__":
    job_generation_timestamp = utils.timestamp(utc=True, tz=False, winsafe=True)

    parser = ArgumentParser(
        'Generate PBS job files; submit with e.g. qsub_wrapper_simple.sh.',
        #formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--template-settings',
        type=str,
        help='Settings file to use for template making.'
    )
    parser.add_argument(
        '--minimizer-settings',
        type=str, required=True,
        help='minimizer settings file.'
    )
    parser.add_argument(
        '--no-alt-fit',
        action='store_true',
        help='No alt hierarchy fit.'
    )
    parser.add_argument(
        '--single-octant',
        action='store_true',
        help='single octant in llh only.'
    )
    parser.add_argument(
        '--numjobs',
        type=int, required=True,
        help='Number of job files to generate.'
    )
    parser.add_argument(
        '--numtrials-per-job',
        type=int, required=True,
        help='Number of LLR trials per job.'
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
        default=16,
        help='Amount of physical memory to request, in GB.')
    parser.add_argument(
        '--vmem',
        type=int,
        default=16,
        help='Amount of virtual to request, in GB.')
    parser.add_argument(
        '--basedir',
        type=str,
        help='''Base directory. Follwing directory strcture is constructed:
Job files are placed in
    <basedir>/jobfiles
Results (json files) are placed in
    <basedir>/llr__<ts basename>__<ms base>/results_rawfiles
Logfiles (*.err, *.out, and PBS *.log -- some or all of which may be
combined) are placed in
    <basedir>/llr__<ts basename>__<ms base>/logfiles
Note that <ts basename>  and <ms basename> are the template and
minimizer settings files' basenames (i.e., without extensions),
respectively.'''
    )
    #parser.add_argument(
    #    '--jobdir',
    #    type=str,
    #    help='Directory for generated job files.'
    #)
    #parser.add_argument(
    #    '--outdir',
    #    type=str, required=True,
    #    help='Directory for llr result files.'
    #)
    #parser.add_argument(
    #    '--logdir',
    #    type=str, required=True,
    #    help='Directory to store PBS, stdout, and stderr log files'
    #)
    args = parser.parse_args()

    args.analysis_script = utils.expandPath(
        '$PISA/pisa/analysis/llr/LLROptimizerAnalysis.py'
    )

    flags = '--save-steps'
    if args.single_octant:
        flags += ' --single_octant'
    if args.no_alt_fit:
        flags += ' --no-alt-fit'
    args.flags = flags

    args.pythonexec_path = os.path.dirname(sys.executable)

    analysis = 'llr'

    # Formulate file names from template and minimizer settings file names,
    # timestamp now, (and later, the file number in this sequence)
    ts_base, _ = os.path.splitext(os.path.basename(args.template_settings))
    ms_base, _ = os.path.splitext(os.path.basename(args.minimizer_settings))
    an_ts_ms_basename = '%s__%s__%s' % (analysis, ts_base, ms_base)

    subdir = an_ts_ms_basename
    batch_basename = '%s__%s' % (an_ts_ms_basename, job_generation_timestamp)

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

    for file_num in xrange(args.numjobs):
        job_basename = '%s__%06d' % (batch_basename, file_num)

        args.jobfile_path = os.path.join(args.jobdir, job_basename + '.pbs')
        args.logfile_path = os.path.join(args.logdir, job_basename + '.log')
        outfile_a_path = os.path.join(args.outdir, job_basename + 'a.json')
        outfile_b_path = os.path.join(args.outdir, job_basename + 'b.json')

        args.command_a = command_template.format(outfile_path=outfile_a_path,
                                                 **vars(args))
        args.command_b = command_template.format(outfile_path=outfile_b_path,
                                                 **vars(args))

        pbs_text = PBS_TEMPLATE.format(**vars(args))

        print 'Writing %d bytes to "%s"' % (len(pbs_text), args.jobfile_path)
        with open(args.jobfile_path, 'w') as jobfile:
            jobfile.write(pbs_text)

    print '\n'
    print 'command_a: %s' % args.command_a
    print 'command_b: %s' % args.command_b
    print ''
    print 'Finished creating %d PBS job files in directory %s' % \
            (args.numjobs, args.jobdir)
